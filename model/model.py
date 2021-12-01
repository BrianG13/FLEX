# -*- coding:utf-8 -*-

import torch
import numpy as np

from model import model_zoo
from model import model_zoo_multi_view
from base.base_model import base_model
from utils import util


class fk_multi_view_model(base_model):
    def __init__(self, config):
        super(fk_multi_view_model, self).__init__()
        print('Multi-View Model')
        self.config = config
        n_joints = self.config.arch.n_joints

        assert len(config.arch.kernel_size) == len(config.arch.stride) == len(config.arch.dilation)
        if n_joints == 20:
            self.parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 8, 12, 13, 14, 8, 16, 17, 18]
        else:
            self.parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.rotation_number = util.ROTATION_NUMBERS.get(config.arch.rotation_type)
        try:
            self.input_feature = 3 if (config.arch.confidence or config.trainer.use_openpose_2d) else 2
        except AttributeError:
            self.input_feature = 3 if config.arch.confidence else 2

        self.n_rotations_predicted = n_joints-5  # Don't predict the rotation of end-effector joint
        self.output_feature = self.n_rotations_predicted*self.rotation_number
        self.output_feature += 2 if config.arch.contact else 0

        if config.trainer.use_3d_world_pose_as_labels is False: # If we are using 3d world as labels only a single prediction for pelvis
            self.output_feature += self.rotation_number * (config.arch.n_views-1)  # Add views-1 rotation pelvis for prediction

        if self.config.arch.translation:  # Add translation prediction for each view
            self.output_feature += config.arch.n_views

        self.BRANCH_S_OUTPUT_FEATURE = 10
        input_features = n_joints*self.input_feature
        input_features += n_joints if 'learnable_with_conf' in self.config.trainer.data else 0
        if not getattr(self.config, 'use_GT_bones', False):  # Option for turning off Branch S
            if getattr(self.config.arch, 'branch_S_multi_view', False):
                self.branch_S = model_zoo_multi_view.pooling_shrink_net(input_features, self.BRANCH_S_OUTPUT_FEATURE, config.arch.kernel_size, config.arch.stride, config.arch.dilation, config.arch.channel, config.arch.stage, config.arch.n_views)
            else:
                self.branch_S = model_zoo.pooling_shrink_net(input_features, self.BRANCH_S_OUTPUT_FEATURE, config.arch.kernel_size, config.arch.stride, config.arch.dilation, config.arch.channel, config.arch.stage)

            if getattr(self.config.trainer, 'optimizer', "") == "adaw":
                self.optimizer_S = torch.optim.AdamW(list(self.branch_S.parameters()))
            else:
                self.optimizer_S = torch.optim.Adam(list(self.branch_S.parameters()), lr=config.trainer.lr, amsgrad=True)

        if not getattr(self.config, 'use_GT_rot', False):  # Option for turning off Branch Q
            print('Early fusion')
            self.branch_Q = model_zoo_multi_view.pooling_net_early_fusion(input_features, self.output_feature,
                                                        config.arch.kernel_size, config.arch.stride, config.arch.dilation,
                                                        config.arch.channel, config.arch.stage, config.arch.kernel_size_stage_1,
                                                        config.arch.kernel_size_stage_2, self.config)
            if getattr(self.config.trainer, 'optimizer', "") == "adaw":
                self.optimizer_Q = torch.optim.AdamW(list(self.branch_Q.parameters()))
            else:
                self.optimizer_Q = torch.optim.Adam(list(self.branch_Q.parameters()), lr=config.trainer.lr, amsgrad=True)

        if config.trainer.use_loss_D:
            self.rotation_D = model_zoo.rotation_D(self.n_rotations_predicted*self.rotation_number, 1, 100, self.n_rotations_predicted)
            self.optimizer_D = torch.optim.Adam(list(self.rotation_D.parameters()), lr=0.0001, amsgrad=True)

        self.fk_layer = model_zoo.fk_layer(config.arch.rotation_type)


        print(f'Branch Q input features: {input_features}')
        print(f'Branch Q output features: {self.output_feature}')
        print('Building the network')

    def forward_S(self, _input):
        return self.branch_S(_input)

    def forward_Q(self, _input):
        return self.branch_Q(_input)[:, :, :12*self.rotation_number]

    def forward_proj(self, _input):
        return self.branch_Q(_input)[:, :, -3] if self.config.arch.contact else self.branch_Q(_input)[:, :, -1]

    def forward_c(self, _input):
        return self.branch_Q(_input)[:, :, -2:]

    def D(self, rotations):
        frame_offset = 5
        delta_input = rotations[:, frame_offset:] - rotations[:, :-frame_offset]
        return self.rotation_D.forward(delta_input)

    def forward_fk(self, _input, norm_parameters, bones_gt, rot_gt):
        starting_pos_totem = getattr(self.config.trainer, 'starting_pos_totem', False)

        if self.config.arch.n_joints == 20:
            complate_indices = np.sort(np.hstack([np.array([0,1,2,4,5,7,8,9,10,12,13,14,16,17,18])*self.rotation_number + i for i in range(self.rotation_number)]))
        else:
            complate_indices = np.sort(np.hstack([np.array([0,1,2,4,5,7,8,9,11,12,14,15])*self.rotation_number + i for i in range(self.rotation_number)]))
        fakes_bones_list = []

        if not getattr(self.config, 'use_GT_bones', False):  # Option for turning off Branch S
            if getattr(self.config.arch, 'branch_S_multi_view', False):
                fake_bones = self.forward_S(_input)
                fakes_bones_list = [fake_bones] * self.config.arch.n_views
            else:
                for i in range(self.config.arch.n_views):
                    fake_bones = self.forward_S(_input[:,i,:,:])
                    fakes_bones_list.append(fake_bones)
        else:
            for i in range(self.config.arch.n_views):
                fakes_bones_list.append(bones_gt[:,i,:,:])

        fake_rotations_list = []
        fake_full_rotations_list = []
        fake_contacts_list = []
        fake_proj_list = [] if self.config.arch.translation else [None]*self.config.arch.n_views

        if not getattr(self.config, 'use_GT_rot', False):  # Option for turning off Branch Q
            output_Q_multi_view = self.branch_Q(_input)
            n_branch_Q_features = self.branch_Q.out_features
            if self.config.arch.contact:
                joints_rotations = output_Q_multi_view[:, :, -self.rotation_number * (self.n_rotations_predicted - 1) - 2:-2]
                contacts = output_Q_multi_view[:, :, -2:]
            elif self.config.arch.translation:
                joints_rotations = output_Q_multi_view[:, :, -self.rotation_number * (self.n_rotations_predicted - 1) - self.config.arch.n_views: -self.config.arch.n_views]
                for i in range(self.config.arch.n_views):
                    fake_proj_list.append(output_Q_multi_view[:, :,
                                          n_branch_Q_features - self.config.arch.n_views + i:n_branch_Q_features - self.config.arch.n_views + i + 1].squeeze())
                contacts = None
            else:
                joints_rotations = output_Q_multi_view[:, :, -self.rotation_number * (self.n_rotations_predicted - 1):]
                contacts = None
            for i in range(self.config.arch.n_views):
                if self.config.trainer.use_3d_world_pose_as_labels:
                    rotation_pelvis = output_Q_multi_view[:,:,:self.rotation_number]
                else:
                    rotation_pelvis = output_Q_multi_view[:,:,i*self.rotation_number:(i+1)*self.rotation_number]
                fake_rotations = torch.cat((rotation_pelvis,joints_rotations), dim=2)
                fake_rotations_list.append(fake_rotations)
                fake_contacts_list.append(contacts)

            for i in range(self.config.arch.n_views):
                fake_rotations_full = torch.zeros((output_Q_multi_view.shape[0], output_Q_multi_view.shape[1],
                                                   self.config.arch.n_joints * self.rotation_number), requires_grad=True)
                if torch.cuda.is_available():
                    fake_rotations_full = fake_rotations_full.cuda()
                fake_rotations_full[:, :, complate_indices] = fake_rotations_list[i].cuda() # torch.Size([64, 196, 68])
                fake_full_rotations_list.append(fake_rotations_full)
        else:
            for i in range(self.config.arch.n_views):
                fake_rotations_list.append(rot_gt[:,i,:,:])
                fake_full_rotations_list.append(rot_gt[:,i,:,:])

        fake_pose_3d_list = []
        for i in range(self.config.arch.n_views):
            if not getattr(self.config, 'use_GT_bones', False):  # Option for turning off Branch S
                skeleton = bones2skel(fakes_bones_list[i].clone().detach(), norm_parameters[2], norm_parameters[3], n_joints=self.config.arch.n_joints,start_position_as_totem=starting_pos_totem)  # torch.Size([#Bsize*frames, 17, 3])
                fake_pose_3d = self.fk_layer.forward(self.parents, skeleton.repeat(_input.shape[2], 1, 1), fake_full_rotations_list[i].contiguous().view(-1, self.config.arch.n_joints, self.rotation_number)).view(_input.shape[0], _input.shape[2], -1)
            else:
                skeleton = GTbones2skel(fakes_bones_list[i].clone().detach(), norm_parameters[2], norm_parameters[3], n_joints=self.config.arch.n_joints,start_position_as_totem=starting_pos_totem) #torch.Size([#Bsize,frames, 17, 3])
                skeleton = skeleton.reshape((skeleton.shape[0]*skeleton.shape[1],skeleton.shape[2],skeleton.shape[3])) #torch.Size([#Bsize*frames, 17, 3])
                fake_pose_3d = self.fk_layer.forward(self.parents, skeleton, fake_full_rotations_list[i].contiguous().view(-1, self.config.arch.n_joints, self.rotation_number)).view(_input.shape[0], _input.shape[2], -1)
            fake_pose_3d_list.append(fake_pose_3d)

        return fakes_bones_list, fake_rotations_list, fake_full_rotations_list,fake_pose_3d_list, fake_contacts_list, fake_proj_list


    def lr_decaying(self, decay_rate):
        optimizer_set = [self.optimizer_length, self.optimizer_rotation]
        for optimizer in optimizer_set:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate


def distance(position1, position2):
    vector = torch.abs(position1 - position2)
    return torch.mean(torch.sqrt(torch.sum(torch.pow(vector, 2), dim=-1)), dim=-1)


def get_velocity(motions, joint_index):
    joint_motion = motions[..., [joint_index*3, joint_index*3 + 1, joint_index*3 + 2]]
    velocity = torch.sqrt(torch.sum((joint_motion[:, 2:] - joint_motion[:, :-2])**2, dim=-1))
    return velocity

def bones2skel(bones, bone_mean, bone_std, n_joints=17, start_position_as_totem=False):
    unnorm_bones = bones * bone_std.unsqueeze(0) + bone_mean.repeat(bones.shape[0], 1, 1)
    skel_in = torch.zeros(bones.shape[0], n_joints, 3)
    if torch.cuda.is_available():
        skel_in = skel_in.cuda()
    if n_joints == 20:
        if start_position_as_totem == True:
            skel_in[:, 1, 0] = -unnorm_bones[:, 0, 0]
            skel_in[:, 4, 0] = unnorm_bones[:, 0, 0]
            skel_in[:, 2, 1] = -unnorm_bones[:, 0, 1]
            skel_in[:, 5, 1] = -unnorm_bones[:, 0, 1]
            skel_in[:, 3, 1] = -unnorm_bones[:, 0, 2]
            skel_in[:, 6, 1] = -unnorm_bones[:, 0, 2]
            skel_in[:, 8, 1] = unnorm_bones[:, 0, 3]
            skel_in[:, 9, 1] = unnorm_bones[:, 0, 4]
            skel_in[:, 12, 1] = unnorm_bones[:, 0, 4]
            skel_in[:, 16, 1] = unnorm_bones[:, 0, 4]
            skel_in[:, 10, 1] = unnorm_bones[:, 0, 5]
            skel_in[:, 11, 1] = unnorm_bones[:, 0, 6]
            skel_in[:, 13, 1] = unnorm_bones[:, 0, 7]
            skel_in[:, 14, 1] = unnorm_bones[:, 0, 8]
            skel_in[:, 15, 1] = unnorm_bones[:, 0, 9]
            skel_in[:, 17, 1] = unnorm_bones[:, 0, 7]
            skel_in[:, 18, 1] = unnorm_bones[:, 0, 8]
            skel_in[:, 19, 1] = unnorm_bones[:, 0, 9]
        else:
            skel_in[:, 1, 0] = -unnorm_bones[:, 0, 0]
            skel_in[:, 4, 0] = unnorm_bones[:, 0, 0]
            skel_in[:, 2, 1] = -unnorm_bones[:, 0, 1]
            skel_in[:, 5, 1] = -unnorm_bones[:, 0, 1]
            skel_in[:, 3, 1] = -unnorm_bones[:, 0, 2]
            skel_in[:, 6, 1] = -unnorm_bones[:, 0, 2]
            skel_in[:, 8, 1] = unnorm_bones[:, 0, 3]
            skel_in[:, 9, 1] = unnorm_bones[:, 0, 4]
            skel_in[:, 12, 1] = unnorm_bones[:, 0, 4]
            skel_in[:, 16, 1] = unnorm_bones[:, 0, 4]
            skel_in[:, 10, 1] = unnorm_bones[:, 0, 5]
            skel_in[:, 11, 1] = unnorm_bones[:, 0, 6]
            skel_in[:, 13, 0] = unnorm_bones[:, 0, 7]
            skel_in[:, 14, 0] = unnorm_bones[:, 0, 8]
            skel_in[:, 15, 0] = unnorm_bones[:, 0, 9]
            skel_in[:, 17, 0] = -unnorm_bones[:, 0, 7]
            skel_in[:, 18, 0] = -unnorm_bones[:, 0, 8]
            skel_in[:, 19, 0] = -unnorm_bones[:, 0, 9]
    else:
        skel_in[:, 1, 0] = -unnorm_bones[:, 0, 0]
        skel_in[:, 4, 0] = unnorm_bones[:, 0, 0]
        skel_in[:, 2, 1] = -unnorm_bones[:, 0, 1]
        skel_in[:, 5, 1] = -unnorm_bones[:, 0, 1]
        skel_in[:, 3, 1] = -unnorm_bones[:, 0, 2]
        skel_in[:, 6, 1] = -unnorm_bones[:, 0, 2]
        skel_in[:, 7, 1] = unnorm_bones[:, 0, 3]
        skel_in[:, 8, 1] = unnorm_bones[:, 0, 4]
        skel_in[:, 9, 1] = unnorm_bones[:, 0, 5]
        skel_in[:, 10, 1] = unnorm_bones[:, 0, 6]
        skel_in[:, 11, 0] = unnorm_bones[:, 0, 7]
        skel_in[:, 12, 0] = unnorm_bones[:, 0, 8]
        skel_in[:, 13, 0] = unnorm_bones[:, 0, 9]
        skel_in[:, 14, 0] = -unnorm_bones[:, 0, 7]
        skel_in[:, 15, 0] = -unnorm_bones[:, 0, 8]
        skel_in[:, 16, 0] = -unnorm_bones[:, 0, 9]

    return skel_in


def GTbones2skel(bones, bone_mean, bone_std,n_joints=17, start_position_as_totem=False):
    unnorm_bones = bones * bone_std.unsqueeze(0) + bone_mean.repeat(bones.shape[0], bones.shape[1], 1)
    skel_in = torch.zeros(bones.shape[0],bones.shape[1], n_joints, 3)
    if torch.cuda.is_available():
        skel_in = skel_in.cuda()

    if n_joints == 20:
        if start_position_as_totem == True:
            skel_in[:,:, 1, 0] = -unnorm_bones[:,:, 0]
            skel_in[:,:, 4, 0] = unnorm_bones[:,:, 0]
            skel_in[:,:, 2, 1] = -unnorm_bones[:,:, 1]
            skel_in[:,:, 5, 1] = -unnorm_bones[:,:, 1]
            skel_in[:,:, 3, 1] = -unnorm_bones[:,:, 2]
            skel_in[:,:, 6, 1] = -unnorm_bones[:,:, 2]
            skel_in[:,:, 8, 1] = unnorm_bones[:,:, 3]
            skel_in[:,:, 9, 1] = unnorm_bones[:,:, 4]
            skel_in[:,:, 12, 1] = unnorm_bones[:,:, 4]
            skel_in[:,:, 16, 1] = unnorm_bones[:,:, 4]
            skel_in[:,:, 10, 1] = unnorm_bones[:,:, 5]
            skel_in[:,:, 11, 1] = unnorm_bones[:,:, 6]
            skel_in[:,:, 13, 1] = unnorm_bones[:,:, 7]
            skel_in[:,:, 14, 1] = unnorm_bones[:,:, 8]
            skel_in[:,:, 15, 1] = unnorm_bones[:,:, 9]
            skel_in[:,:, 17, 1] = unnorm_bones[:,:, 7]
            skel_in[:,:, 18, 1] = unnorm_bones[:,:, 8]
            skel_in[:,:, 19, 1] = unnorm_bones[:,:, 9]
        else:
            skel_in[:,:, 1, 0] = -unnorm_bones[:,:, 0]
            skel_in[:,:, 4, 0] = unnorm_bones[:,:, 0]
            skel_in[:,:, 2, 1] = -unnorm_bones[:,:, 1]
            skel_in[:,:, 5, 1] = -unnorm_bones[:,:, 1]
            skel_in[:,:, 3, 1] = -unnorm_bones[:,:, 2]
            skel_in[:,:, 6, 1] = -unnorm_bones[:,:, 2]
            skel_in[:,:, 8, 1] = unnorm_bones[:,:, 3]
            skel_in[:,:, 9, 1] = unnorm_bones[:,:, 4]
            skel_in[:,:, 12, 1] = unnorm_bones[:,:, 4]
            skel_in[:,:, 16, 1] = unnorm_bones[:,:, 4]
            skel_in[:,:, 10, 1] = unnorm_bones[:,:, 5]
            skel_in[:,:, 11, 1] = unnorm_bones[:,:, 6]
            skel_in[:,:, 13, 0] = unnorm_bones[:,:, 7]
            skel_in[:,:, 14, 0] = unnorm_bones[:,:, 8]
            skel_in[:,:, 15, 0] = unnorm_bones[:,:, 9]
            skel_in[:,:, 17, 0] = -unnorm_bones[:,:, 7]
            skel_in[:,:, 18, 0] = -unnorm_bones[:,:, 8]
            skel_in[:,:, 19, 0] = -unnorm_bones[:,:, 9]
    else:
        skel_in[:,:, 1, 0] = -unnorm_bones[:,:, 0]
        skel_in[:,:, 4, 0] = unnorm_bones[:,:, 0]
        skel_in[:,:, 2, 1] = -unnorm_bones[:,:, 1]
        skel_in[:,:, 5, 1] = -unnorm_bones[:,:, 1]
        skel_in[:,:, 3, 1] = -unnorm_bones[:,:, 2]
        skel_in[:,:, 6, 1] = -unnorm_bones[:,:, 2]
        skel_in[:,:, 7, 1] = unnorm_bones[:,:, 3]
        skel_in[:,:, 8, 1] = unnorm_bones[:,:, 4]
        skel_in[:,:, 9, 1] = unnorm_bones[:,:, 5]
        skel_in[:,:, 10, 1] = unnorm_bones[:,:, 6]
        skel_in[:,:, 11, 0] = unnorm_bones[:,:, 7]
        skel_in[:,:, 12, 0] = unnorm_bones[:,:, 8]
        skel_in[:,:, 13, 0] = unnorm_bones[:,:, 9]
        skel_in[:,:, 14, 0] = -unnorm_bones[:,:, 7]
        skel_in[:,:, 15, 0] = -unnorm_bones[:,:, 8]
        skel_in[:,:, 16, 0] = -unnorm_bones[:,:, 9]
    return skel_in

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))

    return torch.stack((x, y, z), dim=1).view(original_shape)

