import os
import json
import argparse
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import model.model as models
from data.data_loaders import h36m_loader
from utils import util, h36m_utils, Animation, BVH
from model import metric
import matplotlib.gridspec as gridspec
from utils.visualization import show2Dpose, show3Dpose, fig2img
from utils.quaternion import expmap_to_quaternion, qfix, qeuler
from prettytable import PrettyTable

N_JOINTS = 17


def set_3d_to_world_coords(set_3d_array, R_array, T_array):
    set_3d_reshaped = set_3d_array.reshape((set_3d_array.shape[0], set_3d_array.shape[1], -1, 3))
    # Camera Formula: R.T.dot(X.T) + T  (X is the 3d set)
    if torch.cuda.is_available():
        R_T = torch.transpose(R_array, 2, 3).cuda()
        pose_3d_T = torch.transpose(set_3d_reshaped, 2, 3).double().cuda()
        X_cam = torch.matmul(R_T, pose_3d_T).cuda()  # + T_array.cuda()
        X_cam = torch.transpose(X_cam, 2, 3).cuda()
    else:
        R_T = torch.transpose(R_array, 2, 3)
        pose_3d_T = torch.transpose(set_3d_reshaped, 2, 3).double()
        X_cam = torch.matmul(R_T, pose_3d_T)  # + T_array
        X_cam = torch.transpose(X_cam, 2, 3)

    return X_cam


def visualize_2d_and_3d(gt_pose_2d, gt_pose_3d, fake_pose_2d, fake_pose_3d, save_path):
    fig = plt.figure(figsize=(60, 12))

    gs1 = gridspec.GridSpec(2, 2)
    # gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    # plt.axis('off')

    ax0 = plt.subplot(gs1[0, 0])
    show2Dpose(copy.deepcopy(gt_pose_2d), ax0, radius=100)
    ax1 = plt.subplot(gs1[0, 1])
    show2Dpose(copy.deepcopy(fake_pose_2d), ax1, radius=100)
    ax2 = plt.subplot(gs1[1, 0], projection='3d')
    show3Dpose(copy.deepcopy(gt_pose_3d), ax2, radius=600)
    ax3 = plt.subplot(gs1[1, 1], projection='3d')
    show3Dpose(copy.deepcopy(fake_pose_3d), ax3, radius=600)

    if save_path is None:
        fig_img = fig2img(fig)
        plt.close()
        return fig_img
    else:
        plt.show()
        fig.savefig(save_path)


def save_bvh(config, test_data_loader, video_name, pre_proj, poses_2d, pre_rotations_full, pre_bones, test_parameters,
             name_list, output_folder):
    translation = np.zeros((poses_2d.shape[1], 3))
    rotations = pre_rotations_full[0].cpu().numpy()
    length = (pre_bones * test_parameters[3].unsqueeze(0) + test_parameters[2].repeat(pre_bones.shape[0], 1, 1))[
        0].cpu().numpy()
    BVH.save('%s/%s.bvh' % (output_folder, video_name),
             Animation.load_from_network(translation, rotations, length, third_dimension=1), names=name_list)


def main(config, args, output_folder):
    name_list = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine', 'Spine1',
                 'Neck', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']
    name_list_20 = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine',
                    'Spine1', 'Neck', 'Head', 'Site', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
    if config.arch.n_joints == 20:
        name_list = name_list_20

    resume = args.resume
    print(f'Loading checkpoint from: {resume}')
    checkpoint = torch.load(resume)
    config_checkpoint = checkpoint['config']
    print(config_checkpoint)

    model = getattr(models, config.arch.type)(config_checkpoint)

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(model.summary())

    def _prepare_data(device, data, _from='numpy'):
        return torch.from_numpy(np.array(data)).float().to(device) if _from == 'numpy' else data.float().to(device)

    test_data_loader = h36m_loader(config, is_training=False, eval_mode=True)
    test_parameters = [torch.from_numpy(np.array(item)).float().to(device) for item in
                       test_data_loader.dataset.get_parameters()]

    mpjpe_error_list = {}
    acc_error_list = {}

    mpjpe_errors, acc_errors = [], []
    print(f'Evaluating...')
    for video_idx, datas in enumerate(test_data_loader):
        data, video_name = datas[0], datas[1][0]
        data = [[item.float().to(device) for item in view_data] for view_data in data]
        poses_2d_views = torch.stack([view_data[0] for view_data in data], dim=1)
        poses_3d_views = torch.stack([view_data[1] for view_data in data], dim=1)
        bones_views = torch.stack([view_data[2] for view_data in data], dim=1)
        contacts_views = torch.stack([view_data[3] for view_data in data], dim=1)
        alphas_views = torch.stack([view_data[4] for view_data in data], dim=1)
        proj_facters_views = torch.stack([view_data[5] for view_data in data], dim=1)
        root_offsets_views = torch.stack([view_data[6] for view_data in data], dim=1)
        angles_3d_views = torch.stack([view_data[7] for view_data in data], dim=1)
        poses_2d_views_pixels = torch.stack([torch.unsqueeze(view_data[8], 0) for view_data in data], dim=1)

        with torch.no_grad():
            network_output = model.forward_fk(poses_2d_views, test_parameters, bones_views, angles_3d_views)
            fake_bones_views, fake_rotations_views, fake_rotations_full_views, fake_pose_3d_views, fake_c_views, fake_proj_views = network_output[
                                                                                                                                   :6]

        action_name = video_name.split('_')[1].split(' ')[0]  # TODO - REMOVE

        for view_index in range(config.arch.n_views):
            poses_2d_pixels = poses_2d_views_pixels[:, view_index, :, :][0]
            poses_3d = poses_3d_views[:, view_index, :, :]
            alphas = alphas_views[:, view_index, :]

            pre_pose_3d = fake_pose_3d_views[view_index]
            pre_proj = fake_proj_views[view_index]
            pre_rotations_full = fake_rotations_full_views[view_index]
            pre_bones = fake_bones_views[view_index]


            mpjpe_error = metric.mean_points_error(poses_3d, pre_pose_3d) * torch.mean(alphas[0]).data.cpu().numpy()
            accel_error = metric.compute_error_accel(poses_3d, pre_pose_3d, alphas)

            mpjpe_errors.append(mpjpe_error)
            acc_errors.append(accel_error)

            if mpjpe_error and action_name in mpjpe_error_list.keys():
                mpjpe_error_list[action_name].append(mpjpe_error)
                acc_error_list[action_name].append(accel_error)
            else:
                mpjpe_error_list[action_name] = [mpjpe_error]
                acc_error_list[action_name] = [accel_error]

            if args.save_bvh_files:
                save_bvh(config, test_data_loader, video_name + f"_view_{view_index}", pre_proj, poses_2d_pixels,
                         pre_rotations_full, pre_bones, test_parameters, name_list, output_folder)

    error_file = '%s/errors.txt' % output_folder

    with open(error_file, 'w') as f:
        t = PrettyTable(['Action', 'MPJPE (mm)', 'Acc. Error (mm/s^2)'])
        f.writelines("=Action= \t =MPJPE (mm)= \t =Acc. Error(mm/s^2)==")
        for key in mpjpe_error_list.keys():
            mean_pos_error = np.mean(np.array(mpjpe_error_list[key]))
            mean_acc_error = np.mean(np.array(acc_error_list[key]))
            t.add_row([key, f"{mean_pos_error:.2f}", f"{mean_acc_error:.2f}"])
            f.writelines(f'{key} : \t {mean_pos_error:.2f} \t {mean_acc_error:.2f}')

        avg_mpjpe = np.mean(np.array(mpjpe_errors))
        avg_acc_error = np.mean(np.array(acc_errors))
        t.add_row(["", "", ""])
        t.add_row(["Average", f"{avg_mpjpe:.2f}", f"{avg_acc_error:.2f}"])
        f.writelines(f'Total avg. MPJPE: {avg_mpjpe:.2f} \nTotal acc. error: {avg_acc_error:.2f}')
        f.close()
        print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='### MotioNet eveluation')

    parser.add_argument('-r', '--resume', default='./checkpoints/h36m_gt.pth', type=str,
                        help='path to checkpoint (default: None)')
    parser.add_argument('-d', '--device', default="7", type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-i', '--input', default='h36m', type=str, help='h36m or demo or [input_folder_path]')
    parser.add_argument('-o', '--output', default='./output', type=str, help='Output folder')
    parser.add_argument('--save_bvh_files', action='store_true', default=False,
                        help='Flag if save or not the bvh files')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.resume:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        torch.cuda.current_device()
        config = torch.load(args.resume)['config']
    output_folder = util.mkdir_dir('%s/%s' % (args.output, config.trainer.checkpoint_dir.split('/')[-1]))
    print(config)

    main(config, args, output_folder)
