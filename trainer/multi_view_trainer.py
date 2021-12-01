from collections import defaultdict
import numpy as np
import torch
from model import metric
from base.base_trainer import base_trainer
from utils.logger import Logger
from utils import quaternion

from utils import util

LOSS_D_SCALE = 1

class fk_trainer_multi_view(base_trainer):
    def __init__(self, model, resume, config, data_loader, test_data_loader, clearml_logger=None, num_of_views=4):
        super(fk_trainer_multi_view, self).__init__(model, resume, config, logger_path='%s/%s.log' % (
        config.trainer.checkpoint_dir, config.trainer.checkpoint_dir.split('/')[-1]))
        print('Multi-View Trainer')
        self.config = config
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_parameters = [self._prepare_data(item) for item in self.data_loader.dataset.get_parameters()]
        self.test_parameters = [self._prepare_data(item) for item in self.test_data_loader.dataset.get_parameters()]
        self.lambda_s, self.lambda_q, self.lambda_pee, self.lambda_root, self.lambda_f, self.lambda_fc = 0.1, 1, 1.2, 1.3, 0.5, 0.5
        self.steps = 0
        self.train_logger = Logger(
            '%s/%s.log' % (config.trainer.checkpoint_dir, config.trainer.checkpoint_dir.split('/')[-1]))
        self.losses_dict = defaultdict(list)
        self.val_log_dict = defaultdict(list)
        self.clearml_logger = clearml_logger

    def _train_epoch(self, epoch):
        self.model.train()

        print("~~~~~~~~~~ Training ~~~~~~~~~~")
        for batch_idx, data in enumerate(self.data_loader):
            if self.config.trainer.use_loss_D:
                cmu_rotations = data[1]
                cmu_rotations = self._prepare_data(cmu_rotations, _from='tensor')

            data = data[0]
            data = [[self._prepare_data(item, _from='tensor') for item in view_data] for view_data in data]
            poses_2d_views = torch.stack([view_data[0] for view_data in data], dim=1)
            poses_3d_views = torch.stack([view_data[1] for view_data in data], dim=1)
            bones_views = torch.stack([view_data[2] for view_data in data], dim=1)
            contacts_views = torch.stack([view_data[3] for view_data in data], dim=1)
            alphas_views = torch.stack([view_data[4] for view_data in data], dim=1)
            proj_facters_views = torch.stack([view_data[5] for view_data in data], dim=1)
            root_offsets_views = torch.stack([view_data[6] for view_data in data], dim=1)
            angles_3d_views = torch.stack([view_data[7] for view_data in data], dim=1)

            network_output = self.model.forward_fk(poses_2d_views, self.train_parameters, bones_views, angles_3d_views)
            fake_bones_views, fake_rotations_views, fake_rotations_full_views, fake_pose_3d_views, fake_c_views, fake_proj_views = network_output[:6]

            loss_G_GAN, loss_D = 0, 0
            total_loss_bones, total_loss_G, total_loss_D = 0, 0, 0
            for view_index in range(self.config.arch.n_views):
                poses_2d = poses_2d_views[:, view_index, :, :]
                poses_3d = poses_3d_views[:, view_index, :, :]
                bones = bones_views[:, view_index, :, :]
                contacts = contacts_views[:, view_index, :, :]
                alphas = alphas_views[:, view_index, :]
                proj_facters = proj_facters_views[:, view_index, :]
                root_offsets = root_offsets_views[:, view_index, :, :]
                quaternion_angles = angles_3d_views[:, view_index, :, :, :]
                fake_pose_3d = fake_pose_3d_views[view_index]
                fake_rotations = fake_rotations_views[view_index]
                fake_c = fake_c_views[view_index]
                fake_proj = fake_proj_views[view_index]

                if not getattr(self.config, 'use_GT_bones', False):  # Option flag to turn OFF Branch S
                    fake_bones = fake_bones_views[view_index]
                else:
                    fake_bones = None

                if not getattr(self.config, 'use_GT_rot', False):  # Option flag to turn OFF Branch Q
                    fake_rotations_full = fake_rotations_full_views[view_index]
                else:
                    fake_rotations_full = quaternion_angles

                loss_bones, loss_G, loss_D, loss_dict = self.calc_loss_for_single_output(fake_bones, fake_rotations,
                                                                                         fake_rotations_full,
                                                                                         fake_pose_3d, fake_c,
                                                                                         fake_proj, bones, poses_3d,
                                                                                         contacts, proj_facters,
                                                                                         quaternion_angles)
                scale = 1 / self.config.arch.n_views
                total_loss_bones += scale * loss_bones
                total_loss_G += scale * loss_G
                total_loss_D += scale * loss_D

                iteration = len(self.data_loader) * (epoch - 1) + batch_idx + 1 # + permutation_idx
                if self.clearml_logger:
                    train_mpjpe = metric.mean_points_error(fake_pose_3d, poses_3d) * torch.mean(alphas[0]).data.cpu().numpy()
                    self.clearml_logger.report_scalar("Train MPJPE", f'View {view_index}', iteration=iteration, value=train_mpjpe)
                    for loss_name in loss_dict.keys():
                        loss_val = loss_dict[loss_name]
                        if loss_val != 0:
                            self.clearml_logger.report_scalar("Losses", f'{loss_name}_{view_index}',
                                                              iteration=iteration, value=loss_val)

            if self.config.trainer.use_loss_D:
                self.model.optimizer_D.zero_grad()
                total_loss_D.backward()
                self.model.optimizer_D.step()
            if not getattr(self.config, 'use_GT_bones', False):  # Option flag to turn OFF Branch S
                self.model.optimizer_S.zero_grad()
                total_loss_bones.backward()
                self.model.optimizer_S.step()
            if not getattr(self.config, 'use_GT_rot', False):  # Option flag to turn OFF Branch Q
                self.model.optimizer_Q.zero_grad()
                total_loss_G.backward()
                self.model.optimizer_Q.step()

            train_log = {'loss_G': total_loss_G, 'loss_bones': total_loss_bones, 'loss_D': total_loss_D,
                         'loss_G_GAN': loss_G_GAN}

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.writer.set_step(self.steps, mode='train')
                self.writer.set_scalars(train_log)
                training_message = f'Train Epoch: {epoch} [{self.data_loader.batch_size * batch_idx}/{self.data_loader.n_samples} ({100.0 * batch_idx / len(self.data_loader):.0f})%]\t'
                for key, value in train_log.items():
                    if value > 0:
                        training_message += '{}: {:.6f}\t'.format(key, value)
                self.train_logger.info(training_message)
            self.steps += 1

            # Document losses
            for loss_key in loss_dict.keys():
                if isinstance(loss_dict[loss_key], int) or isinstance(loss_dict[loss_key], float):
                    self.losses_dict[loss_key].append(loss_dict[loss_key])
                else:
                    self.losses_dict[loss_key].append(loss_dict[loss_key].item())

        print("~~~~~~~~~~ Validation ~~~~~~~~~~")
        val_log = self._valid_epoch(epoch)
        for val_log_key, val_log_value in val_log.items():
            self.val_log_dict[val_log_key].append(val_log_value)
        return val_log

    def set_3d_to_world_coords(self, set_3d_array, R_array, T_array):
        set_3d_reshaped = set_3d_array.reshape((set_3d_array.shape[0], set_3d_array.shape[1], -1, 3))
        # Camera Formula: R.T.dot(X.T) + T  (X is the 3d set)
        if torch.cuda.is_available():
            X_cam = torch.matmul(torch.transpose(R_array, 2, 3).cuda(),
                                 torch.transpose(set_3d_reshaped, 2, 3).double().cuda()).cuda()  # + T_array.cuda()
            X_cam = torch.transpose(X_cam, 2, 3).cuda()
        else:
            X_cam = torch.matmul(torch.transpose(R_array, 2, 3),
                                 torch.transpose(set_3d_reshaped, 2, 3).double())  # + T_array
            X_cam = torch.transpose(X_cam, 2, 3)

        return X_cam

    def calc_loss_for_single_output(self, fake_bones, fake_rotations, fake_rotations_full, fake_pose_3d, fake_c,
                                    fake_proj, bones,
                                    poses_3d, contacts, proj_facters, rotations_full):

        def get_velocity(motions, joint_index):
            joint_motion = motions[..., [joint_index * 3, joint_index * 3 + 1, joint_index * 3 + 2]]
            velocity = torch.sqrt(torch.sum((joint_motion[:, 2:] - joint_motion[:, :-2]) ** 2, dim=-1))
            return velocity

        loss_bones, loss_angles = 0, 0
        if fake_bones is not None:
            loss_bones = torch.mean(torch.norm(fake_bones - bones, dim=-1))
        position_weights = torch.ones((1, self.config.arch.n_joints)).cuda()
        if self.config.arch.n_joints == 20:
            position_weights[:, [0, 3, 6, 11, 15, 19]] = getattr(self.config.trainer, 'arms_and_legs_weight',
                                                                 self.lambda_pee)
        else:
            position_weights[:, [0, 3, 6, 8, 13, 17]] = self.lambda_pee

        complate_indices = np.sort(
            np.hstack([np.array([0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18]) * 4 + i for i in range(4)]))
        rotations = rotations_full[:, :, 0, complate_indices]

        temp_fake_3d = fake_pose_3d.contiguous().view((-1, self.config.arch.n_joints, 3))

        temp_3d = poses_3d.contiguous().view((-1, self.config.arch.n_joints, 3))

        loss_positions = torch.mean(
            torch.norm((temp_fake_3d - temp_3d), dim=-1) * position_weights) if self.config.trainer.use_loss_3d else 0
        loss_root = torch.mean(torch.norm(fake_proj - proj_facters, dim=-1)) if self.config.arch.translation else 0
        loss_f = torch.mean(torch.norm(fake_c - contacts, dim=-1)) if self.config.trainer.use_loss_foot else 0
        loss_fc = (torch.mean(get_velocity(fake_pose_3d, 3)[contacts[:, 1:-1, 0] == 1] ** 2) + torch.mean(
            get_velocity(fake_pose_3d, 6)[contacts[:, 1:-1, 0] == 1] ** 2)) if self.config.trainer.use_loss_foot else 0
        if self.config.trainer.use_loss_D:
            loss_G_GAN, loss_D = self.calc_discriminator_loss(rotations, fake_rotations)
            loss_D = loss_D * LOSS_D_SCALE
        else:
            loss_G_GAN, loss_D = 0, 0

        loss_G = loss_positions + loss_root * self.lambda_root + loss_f * self.lambda_f + loss_fc * self.lambda_fc + loss_G_GAN * self.lambda_q
        loss_D = loss_D * self.lambda_q

        train_log = {'loss_G': loss_G, 'loss_positions': loss_positions, 'loss_bones': loss_bones,
                     'lambda_root': loss_root, 'loss_f': loss_f, 'loss_fc': loss_fc,
                     'loss_G_GAN': loss_G_GAN, 'loss_D': loss_D}

        return loss_bones, loss_G, loss_D, train_log

    def calc_discriminator_loss(self, rotations, fake_rotations):
        G_real = self.model.D(fake_rotations.detach())
        D_real = self.model.D(rotations)
        D_fake = self.model.D(fake_rotations.detach())
        loss_G_GAN = torch.mean(torch.norm((G_real - 1) ** 2, dim=-1))
        loss_D = torch.mean(torch.norm((D_real - 1) ** 2)) + torch.mean(torch.sum((D_fake) ** 2, dim=-1))
        return loss_G_GAN, loss_D

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_metrics_list = [0] * self.config.arch.n_views
        total_val_loss_list = [0] * self.config.arch.n_views
        for batch_idx, datas in enumerate(self.test_data_loader):
            data, video_name, video_and_frame = datas
            data = [[self._prepare_data(item, _from='tensor') for item in view_data] for view_data in data]

            poses_2d_views = torch.stack([view_data[0] for view_data in data], dim=1)
            poses_3d_views = torch.stack([view_data[1] for view_data in data], dim=1)
            bones_views = torch.stack([view_data[2] for view_data in data], dim=1)
            contacts_views = torch.stack([view_data[3] for view_data in data], dim=1)
            alphas_views = torch.stack([view_data[4] for view_data in data], dim=1)
            proj_facters_views = torch.stack([view_data[5] for view_data in data], dim=1)
            root_offsets_views = torch.stack([view_data[6] for view_data in data], dim=1)
            angles_3d_views = torch.stack([view_data[7] for view_data in data], dim=1)
            with torch.no_grad():
                network_output = self.model.forward_fk(poses_2d_views, self.test_parameters, bones_views,
                                                       angles_3d_views)

            fake_bones_views, fake_rotations_views, fake_rotations_full_views, fake_pose_3d_views, fake_c_views, fake_proj_views = network_output[
                                                                                                                                   :6]

            for view_index in range(self.config.arch.n_views):
                poses_3d = poses_3d_views[:, view_index, :, :]
                alphas = alphas_views[:, view_index, :]
                angles_3d = angles_3d_views[:, view_index, :, :, :]
                quaternion_angles = angles_3d  # util.euler_to_quaternions(angles_3d.cpu().numpy(), order = 'zxy')
                fake_pose_3d = fake_pose_3d_views[view_index]
                poses_2d = poses_2d_views[:, view_index, :, :]
                mpjpe = metric.mean_points_error(fake_pose_3d, poses_3d) * torch.mean(alphas[0]).data.cpu().numpy()
                total_val_metrics_list[view_index] += mpjpe
                total_val_loss_list[view_index] += metric.mean_points_error(fake_pose_3d,
                                                                            poses_3d)
            for view_index in range(self.config.arch.n_views):
                len_data_loader = len(self.test_data_loader)
                val_log = {'val_metric': total_val_metrics_list[view_index] / len_data_loader,
                           'val_loss': total_val_loss_list[view_index] / len_data_loader}

                iteration = len(self.data_loader) * epoch
                if self.clearml_logger:
                    self.clearml_logger.report_scalar("Test MPJPE", f'View {view_index}', iteration=iteration,
                                                      value=val_log['val_metric'])
                    self.clearml_logger.report_scalar("Losses", f'Loss View {view_index}', iteration=iteration,
                                                      value=val_log['val_loss'])

            self.writer.set_step(epoch, mode='valid')
            self.writer.set_scalars(val_log)

        self.train_logger.info('Evaluation: mean_points_error: {:.6f} loss: {:.6f}'.format(val_log['val_metric'], val_log['val_loss']))

        return val_log
