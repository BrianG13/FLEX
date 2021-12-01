# -*- coding:utf-8 -*-
import os
import csv
import copy
import itertools
import random
import numpy as np

from torch.utils.data import Dataset
from utils import h36m_utils, util
from utils.visualization import show2Dpose, show3Dpose, fig2img
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_all_views(poses_2d_list, save_path):
    n_views = len(poses_2d_list)
    fig = plt.figure(figsize=(120, 24))
    gs1 = gridspec.GridSpec(n_views, 5)

    for view_idx in range(n_views):
        for i in range(5):
            ax0 = plt.subplot(gs1[view_idx, i])
            show2Dpose(copy.deepcopy(poses_2d_list[view_idx][1740 + 20 * i]), ax0, radius=300)
            ax0.set_ylim(ax0.get_ylim()[::-1])

    plt.tight_layout()

    if save_path is None:
        fig_img = fig2img(fig)
        plt.close()
        return fig_img
    else:
        # plt.show()
        fig.savefig(save_path)


def plot2DPose(poses_2d, img_path=None, save_path=None):
    from matplotlib import image
    lcolor = "#3498db"
    rcolor = "#e74c3c"
    vals = poses_2d.reshape((-1, 2))
    if vals.shape[0] == 14:
        I = np.array([6, 5, 4, 3, 2, 12, 11, 10, 9, 8, 4, 3, 13]) - 1
        J = np.array([5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 10, 9, 14]) - 1
    else:
        I = np.array([0, 1, 2, 0, 4, 5, 0, 8, 9, 10, 9, 13, 14, 9, 17, 18])
        J = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    fig = plt.figure(figsize=(48, 48))
    gs1 = gridspec.GridSpec(1, 1)
    # plt = plt.subplot(gs1[0, 0])
    # Make connection matrix

    ax = plt.gca()

    for i in np.arange(len(I)):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        if np.mean(x) != 0 and np.mean(y) != 0:
            ax.plot(x, y, lw=10, c=lcolor if LR[i] else rcolor)

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    RADIUS = 400  # space around the subject
    xroot, yroot = vals[0, 0], vals[0, 1]
    ax.set_xlim([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim([-RADIUS + yroot, RADIUS + yroot])
    if True:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    # plt.aspect('equal')
    # for i in range(vals.shape[0]):
    #     plt.text(vals[i, 0], vals[i, 1], f'{i}', color='black')

    for i in range(vals.shape[0]):
        ax.add_patch(plt.Circle((vals[i, 0], vals[i, 1]), 5, color='w', edgecolor='k'))

    if img_path is not None:
        img = image.imread(img_path)
        plt.imshow(img)

    ax.set_ylim(ax.get_ylim()[::-1])

    if save_path:
        fig.savefig(save_path)


def plot3DPose(poses_3d, save_path=None, view_angle=30):
    lcolor = "#3498db"
    rcolor = "#e74c3c"
    if poses_3d.size == 51:
        # I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        # J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        # LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        I = [0, 1, 2, 5, 4, 3, 6, 7, 8, 9, 8, 11, 10, 8, 13, 14]
        J = [1, 2, 6, 4, 3, 6, 7, 8, 16, 16, 12, 12, 11, 13, 14, 15]
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    else:
        I = np.array([0, 1, 2, 0, 4, 5, 0, 8, 9, 10, 9, 13, 14, 9, 17, 18])
        J = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19])
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    fig = plt.figure(figsize=(60, 60))
    gs1 = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs1[0, 0], projection='3d')
    vals = poses_3d.reshape((-1, 3))

    # vals[:, [1, 2]] = vals[:, [2, 1]]
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=14, c=lcolor if LR[i] else rcolor)

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    # ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    # ax.set_ylim3d([RADIUS + yroot, -RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_zticklabels([])
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    # for i in range(JOINTS):
    #     plt.text(vals[i, 0], vals[i, 1],vals[i, 2], f'{i}', color='black')
    for i in range(vals.shape[0]):
        ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], s=500, c='black')
        # ax.add_patch(plt.Circle((vals[i, 0], vals[i, 1],vals[i, 2]), 0.2, color='black'))

    ax.view_init(10, view_angle)

    if save_path:
        fig.savefig(save_path)


def read_file(path):
    '''
    Read an individual file in expmap format,
    and return a NumPy tensor with shape (sequence length, number of joints, 3).
    '''
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    data = np.array(data, dtype='float64')
    return data.reshape(data.shape[0], -1, 3)


def createDict(*args):
    return dict(((k, eval(k)) for k in args))


def compare_2d(gt_pose_2d, fake_pose_2d, save_path, title):
    fig = plt.figure(figsize=(60, 12))
    fig.suptitle(title, fontsize=32)
    gs1 = gridspec.GridSpec(2, 5)

    for i in range(5):
        ax0 = plt.subplot(gs1[0, i])
        show2Dpose(copy.deepcopy(gt_pose_2d[i * 200]), ax0, radius=200)
        ax0.set_ylim(ax0.get_ylim()[::-1])
        ax1 = plt.subplot(gs1[1, i])
        show2Dpose(copy.deepcopy(fake_pose_2d[i * 200]), ax1, lcolor="#000000", rcolor="#000000", radius=200)

    if save_path is None:
        fig_img = fig2img(fig)
        plt.close()
        return fig_img
    else:
        # plt.show()
        fig.savefig(save_path)


class multi_view_h36_dataset(Dataset):

    def __init__(self, config, is_train=True, num_of_views=4, fake_cam=None, eval_mode=False):
        print('MultiView H36M Data')
        BASE_DATA_PATH = os.path.dirname(os.path.realpath(__file__))

        poses_3d, poses_2d, bones, alphas, contacts, proj_facters = [], [], [], [], [], []
        poses_2d_pixels = []
        root_offsets, angles_3d = [], []
        self.cameras = h36m_utils.load_cameras(f'{BASE_DATA_PATH}/cameras.h5')
        self.complate_indices = np.sort(
            np.hstack([np.array([0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18]) * 4 + i for i in range(4)]))

        self.frame_numbers = []
        self.video_and_frame = []
        self.video_name = []
        self.config = config
        self.eval_mode = eval_mode
        self.is_train = is_train
        subjects = h36m_utils.TRAIN_SUBJECTS if is_train else h36m_utils.TEST_SUBJECTS

        self.fake_cam = fake_cam

        if config.arch.n_joints == 20:
            self.positions_set = np.load(f'{BASE_DATA_PATH}/new_data_h36m.npz', allow_pickle=True)['positions_3d'].item()
            self.angles_set = np.load(f'{BASE_DATA_PATH}/new_data_h36m_angles.npz', allow_pickle=True)['angles_3d'].item()
        else:
            self.positions_set = np.load(f'{BASE_DATA_PATH}/data_h36m.npz', allow_pickle=True)['positions_3d'].item()
            self.angles_set = np.load(f'{BASE_DATA_PATH}/data_h36m_angles.npz', allow_pickle=True)['angles_3d'].item()

        if config.trainer.data == 'cpn':
            print('USING CPN 2D POSES')
            self.positions_set_2d = np.load(f'{BASE_DATA_PATH}/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)['positions_2d'].item()
        elif config.trainer.data == 'detectron':
            self.positions_set_2d = np.load(f'{BASE_DATA_PATH}/data_2d_h36m_detectron_ft_h36m.npz', allow_pickle=True)[
                'positions_2d'].item()
        elif config.trainer.data == 'learnable':
            self.positions_set_2d = self.load_learnable_2d_backbone_positions(BASE_DATA_PATH, is_train=is_train)
        elif config.trainer.data == 'learnable_with_conf_distorted':
            self.positions_set_2d = self.load_learnable_2d_backbone_positions(BASE_DATA_PATH, is_train=is_train,
                                                                              with_conf=True, distorted=True)
        elif config.trainer.data == 'learnable_with_conf_undistorted':
            self.positions_set_2d = self.load_learnable_2d_backbone_positions(BASE_DATA_PATH, is_train=is_train,
                                                                              with_conf=True, distorted=False)
        elif config.trainer.data == 'openpose':
            self.positions_set_2d = np.load(f'{BASE_DATA_PATH}/data/h36m_openpose_2d_positions.npz', allow_pickle=True)['positions_2d_openpose'].item()

        self.n_camera_views = num_of_views

        for subject in subjects:
            for action in self.positions_set['S%s' % subject].keys():
                action_sequences = self.positions_set['S%s' % subject][action]
                angles_sequences = self.angles_set['S%s' % subject][action]
                if getattr(config.trainer, 'train_only_on_cameras', None) is not None:
                    camera_views_combination = config.trainer.train_only_on_cameras
                else:
                    camera_views_combination = list(itertools.combinations(range(len(action_sequences)), self.n_camera_views))
                for views_camera_indexes in camera_views_combination:
                    print(f'Processing Subject: S{subject} , Action: {action} , Cameras: {views_camera_indexes}')
                    set_views_poses_3d, set_views_poses_2d = [], []
                    set_views_bones, set_views_alphas, set_views_contacts, set_views_proj_facters = [], [], [], []
                    set_views_root_offsets, set_views_angles = [], []
                    set_views_poses_2d_pixels = []
                    for c_idx in views_camera_indexes:
                        set_3d = action_sequences[c_idx]
                        set_angles_3d = angles_sequences[c_idx]
                        set_angles_3d = set_angles_3d[:, :, [2, 0, 1]]
                        set_q_rot_3d = util.euler_to_quaternions(set_angles_3d, order='zxy').cpu().numpy()
                        set_root_offset, set_3d, set_3d_world, set_2d, set_bones, set_alphas, set_contacts, set_proj_facters, set_2d_original = self.process_data_for_subject_camera_3d_pose(config, subject, action, c_idx, set_3d)
                        set_views_poses_3d.append(set_3d)
                        set_views_poses_2d.append(set_2d)
                        set_views_poses_2d_pixels.append(set_2d_original)
                        set_views_bones.append(set_bones)
                        set_views_alphas.append(set_alphas)
                        set_views_contacts.append(set_contacts)
                        set_views_proj_facters.append(set_proj_facters)
                        set_views_root_offsets.append(set_root_offset)
                        set_views_angles.append(set_q_rot_3d)


                    set_views_poses_3d = np.stack(set_views_poses_3d, axis=0)
                    set_views_poses_2d = np.stack(set_views_poses_2d, axis=0)
                    set_views_poses_2d_pixels = np.stack(set_views_poses_2d_pixels, axis=0)
                    set_views_bones = np.stack(set_views_bones, axis=0)
                    set_views_proj_facters = np.stack(set_views_proj_facters, axis=0)
                    set_views_alphas = np.stack(set_views_alphas, axis=0)
                    set_views_contacts = np.stack(set_views_contacts, axis=0)
                    set_views_root_offsets = np.stack(set_views_root_offsets, axis=0)
                    set_views_angles = np.stack(set_views_angles, axis=0)

                    subject_video_camera_name = 'S%s_%s_%s' % (subject, action, "c" + '_'.join([str(x) for x in views_camera_indexes]))

                    for permutation in [list(range(self.n_camera_views))]:  # hack for a single permutation
                        permutation = list(permutation)

                        self.frame_numbers.append(set_3d.shape[0])
                        self.video_name.append(subject_video_camera_name)
                        self.video_and_frame.extend([subject_video_camera_name + f"_frame_{i}" for i in range(set_3d.shape[0])])

                        if self.n_camera_views == 1:
                            permutation = [0]

                        poses_3d.append(set_views_poses_3d[permutation, :, :])
                        poses_2d.append(set_views_poses_2d[permutation, :, :])
                        poses_2d_pixels.append(set_views_poses_2d_pixels[permutation, :, :])
                        bones.append(set_views_bones[permutation, :, :])
                        proj_facters.append(set_views_proj_facters[permutation, :])
                        alphas.append(set_views_alphas[permutation, :])
                        contacts.append(set_views_contacts[permutation, :, :])

                        root_offsets.append(set_views_root_offsets[permutation, :, :])
                        angles_3d.append(set_views_angles[permutation, :, :])

        print('Concatenate.. ')
        # poses_3d is a list each item: is a [#Views, #FramesSingleVideo, 17*3]
        self.poses_3d = np.concatenate(poses_3d, axis=1)
        self.angles_3d = np.concatenate(angles_3d, axis=1)
        self.poses_2d = np.concatenate(poses_2d, axis=1)
        self.poses_2d_pixels = np.concatenate(poses_2d_pixels, axis=1)
        self.proj_facters = np.concatenate(proj_facters, axis=1)
        self.contacts = np.concatenate(contacts, axis=1)
        self.alphas = np.concatenate(alphas, axis=1)
        self.bones = np.concatenate(bones, axis=1)
        self.root_offsets = np.concatenate(root_offsets, axis=1)
        self.video_and_frame = np.asarray(self.video_and_frame)

        if is_train:
            print('Loading CMU data for Discriminator')
            rotations_set = np.load(f'{BASE_DATA_PATH}/data_cmu.npz', allow_pickle=True)['rotations']
            self.r_frame_numbers = [r_array.shape[0] for r_array in rotations_set]
            self.rotations = np.concatenate(rotations_set, axis=0)
            self.rotations = self.rotations.reshape((self.rotations.shape[0], -1))
            print('Done Loading CMU data for Discriminator')

        if config.arch.confidence:
            self.poses_2d_noised, confidence_maps = self.add_noise(self.poses_2d, training=is_train)
            self.poses_2d_noised_with_confidence = np.zeros(
                (self.poses_2d_noised.shape[0], int(self.poses_2d_noised.shape[-1] / 2 * 3)))
            for joint_index in range(int(self.poses_2d_noised.shape[-1] / 2)):
                self.poses_2d_noised_with_confidence[:, 3 * joint_index] = self.poses_2d_noised[:, 2 * joint_index]
                self.poses_2d_noised_with_confidence[:, 3 * joint_index + 1] = self.poses_2d_noised[:,
                                                                               2 * joint_index + 1]
                self.poses_2d_noised_with_confidence[:, 3 * joint_index + 2] = (confidence_maps[:,
                                                                                2 * joint_index] + confidence_maps[:,
                                                                                                   2 * joint_index]) / 2


        print('Reshaping.. ')
        self.poses_2d = self.poses_2d.reshape((-1, self.poses_2d.shape[-1]))
        self.bones = self.bones.reshape((-1, self.bones.shape[-1]))
        self.proj_facters = self.proj_facters.reshape((-1))

        print('Normalizing Poses 2D ..')
        self.poses_2d, self.poses_2d_mean, self.poses_2d_std = util.normalize_data(self.poses_2d_noised_with_confidence if config.arch.confidence else self.poses_2d)
        print('Normalizing Bones ..')
        self.bones, self.bones_mean, self.bones_std = util.normalize_data(self.bones)
        print('Normalizing Proj. Factors ..')
        self.proj_facters, self.proj_mean, self.proj_std = util.normalize_data(self.proj_facters)

        self.poses_2d = self.poses_2d.reshape((self.n_camera_views, -1, self.poses_2d.shape[-1]))
        self.bones = self.bones.reshape((self.n_camera_views, -1, self.bones.shape[-1]))
        self.proj_facters = self.proj_facters.reshape((self.n_camera_views, -1))

        # all_poses_2d_permutations = []
        # all_bones_permutations = []
        # all_proj_facters_permutations = []
        # for permutation in list(permutations(range(self.n_camera_views), self.n_camera_views)):
        #     all_poses_2d_permutations.append(self.poses_2d[permutation, :, :])

        self.set_sequences()

        self.translate_to_lists()

    def load_learnable_2d_backbone_positions(self, base_data_path, is_train=False, with_conf=False, distorted=True):
        def fix_action_title(self, subject, action_name, n_frames):
            if subject == 'S11' and action_name == 'Directions-1':
                return 'Directions 1'
            scenario_name = action_name.split('-')[0]
            scenario_name = 'Photo' if scenario_name == 'TakingPhoto' else scenario_name
            scenario_name = 'WalkDog' if scenario_name == 'WalkingDog' else scenario_name
            scenario_name = 'WalkTogether' if scenario_name == 'WalkingTogether' else scenario_name

            if scenario_name in self.positions_set[subject]:
                first_action = scenario_name
            elif scenario_name + " 1" in self.positions_set[subject]:
                first_action = scenario_name + " 1"
            elif scenario_name + " 2" in self.positions_set[subject]:
                first_action = scenario_name + " 2"

            if scenario_name + " 1" in self.positions_set[subject] and scenario_name + " 1" != first_action:
                second_action = scenario_name + " 1"
            elif scenario_name + " 2" in self.positions_set[subject] and scenario_name + " 2" != first_action:
                second_action = scenario_name + " 2"
            elif scenario_name + " 3" in self.positions_set[subject]:
                second_action = scenario_name + " 3"
            print(first_action)

            if abs(self.positions_set[subject][first_action][0].shape[0] - n_frames) < abs(
                    self.positions_set[subject][second_action][0].shape[0] - n_frames):
                return first_action
            else:
                return second_action

        JOINTS_INDEX_MAPPING = [6, 2, 1, 0, 3, 4, 5, 6, 7, 8, 16, 9, 8, 13, 14, 15, 8, 12, 11, 10]
        #				        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        positions_set = {}
        if is_train:
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:
            subjects = ['S9', 'S11']
        file_prefix = 'data_h36m_keypoints_2d_from_backbone'
        matching_dir = {}
        file_suffix = ""
        if with_conf and distorted:
            file_suffix = "_with_conf_distorted"
        if with_conf and not distorted:
            file_suffix += "_with_conf_undistorted"
        for sub in subjects:
            matching_dir[sub] = {}
            print(f'Loading 2D from learnable trig. backbone for: {sub}')
            positions_set[sub] = {}
            if with_conf:
                file_path = f'{base_data_path}/{file_prefix}_{sub}{file_suffix}.npz'
            else:
                file_path = f'{base_data_path}/{file_prefix}_{sub}.npz'

            data = np.load(file_path, allow_pickle=True)['data_2d_backbone'].item()
            for action_name, poses_2d_data in data.items():
                if sub == 'S11' and action_name == 'Directions-2':
                    continue

                n_frames = len(poses_2d_data.keys())
                temp_action_name = action_name
                action_name = fix_action_title(self, sub, action_name, n_frames)
                print(f'{temp_action_name} -> {action_name}')
                matching_dir[sub][temp_action_name] = action_name
                positions_set[sub][action_name] = []
                for cam_idx in range(4):
                    if with_conf:
                        positions_set[sub][action_name].append(np.zeros((n_frames, 20, 3)))
                    else:
                        positions_set[sub][action_name].append(np.zeros((n_frames, 20, 2)))
                    for frame_idx in range(n_frames):
                        frame_data = poses_2d_data[frame_idx]
                        if with_conf:
                            positions_set[sub][action_name][cam_idx][frame_idx, :, :2] = frame_data[0][0, cam_idx,
                                                                                         JOINTS_INDEX_MAPPING, :]
                            positions_set[sub][action_name][cam_idx][frame_idx, :, 2] = frame_data[1][
                                0, cam_idx, JOINTS_INDEX_MAPPING]
                        else:
                            positions_set[sub][action_name][cam_idx][frame_idx, :, :] = frame_data[0, cam_idx,
                                                                                        JOINTS_INDEX_MAPPING, :]
        return positions_set

    def translate_to_lists(self):
        self.poses_3d_list, self.poses_2d_list, self.bones_list, self.alphas_list, self.proj_facters_list = [], [], [], [], []
        self.contacts_list, self.angles_3d_list, self.root_offsets_list = [], [], []
        for i in range(self.n_camera_views):
            self.poses_3d_list.append(self.poses_3d[i])
            self.poses_2d_list.append(self.poses_2d[i])
            self.bones_list.append(self.bones[i])
            self.alphas_list.append(self.alphas[i])
            self.proj_facters_list.append(self.proj_facters[i])
            self.contacts_list.append(self.contacts[i])
            self.angles_3d_list.append(self.angles_3d[i])
            self.root_offsets_list.append(self.root_offsets[i])

    def __getitem__(self, index):
        items_index = self.sequence_index[index]
        # Start of experiment"
        views_data = []
        for i in range(self.n_camera_views):
            poses_3d = self.poses_3d_list[i][items_index]
            poses_2d = self.poses_2d_list[i][items_index]
            bones = self.bones_list[i][items_index]
            alphas = self.alphas_list[i][items_index]
            proj_facters = self.proj_facters_list[i][items_index]
            contacts = self.contacts_list[i][items_index]
            angles_3d = self.angles_3d_list[i][items_index]
            root_offsets = self.root_offsets_list[i][items_index]
            poses_2d_pixels = self.poses_2d_pixels[i][items_index]

            views_data.append(
                [poses_2d, poses_3d, bones, contacts, alphas, proj_facters, root_offsets, angles_3d, poses_2d_pixels])
        video_and_frames = self.video_and_frame[items_index].tolist()
        first_frame_idx = int(video_and_frames[0].split('_')[-1])
        last_frame_idx = int(video_and_frames[-1].split('_')[-1])
        subject_and_video_title = f"{video_and_frames[0].split('_')[0]}_{video_and_frames[0].split('_')[1]}"
        title_with_frame_range = f'{subject_and_video_title}_from_{first_frame_idx}_to_{last_frame_idx}'
        if self.is_train:
            rotations = self.rotations[self.r_sequence_index[np.array(index) % self.r_sequence_index.shape[0]]]
            return views_data, rotations, title_with_frame_range
        else:
            return views_data, self.video_name[index % len(self.video_name)], title_with_frame_range

    def __len__(self):
        return self.sequence_index.shape[0]

    def get_video_and_frame_details(self, index):
        items_index = self.sequence_index[index]
        return self.video_and_frame[items_index]

    def process_data_for_subject_camera_3d_pose(self, config, subject, action, c_idx, set_3d):
        augment_depth = random.randint(-5, 20) if config.trainer.data_aug_depth else 0

        set_3d = set_3d.copy().reshape((set_3d.shape[0], -1))
        R, T, f, c, k, p, res_w, res_h = self.cameras[(subject, c_idx)]

        set_3d_world = h36m_utils.camera_to_world_frame(set_3d.reshape((-1, 3)), R, T).reshape(set_3d.shape)

        if self.fake_cam and c_idx == 0:
            R, T, f, c, k, p, res_w, res_h = self.fake_cam
            set_3d = h36m_utils.world_to_camera_frame(set_3d_world.reshape((-1, 3)), R, T).reshape(set_3d_world.shape)

        set_gt = \
            h36m_utils.project_2d(set_3d.reshape((-1, 3)), R, T, f, c, k, p, augment_depth=augment_depth, from_world=False)[
                0].reshape((set_3d.shape[0], int(set_3d.shape[-1] / 3 * 2)))
        if config.trainer.data == 'gt':
            set_2d = set_gt
        else:
            set_2d = self.positions_set_2d['S%s' % subject][action][c_idx]
            min_length = min(set_3d.shape[0], set_2d.shape[0])
            if config.trainer.data == 'learnable_with_conf':
                set_2d_conf = set_2d[:, :, 2]
                set_2d = set_2d[:, :, :2]
            set_2d = set_2d.reshape((set_2d.shape[0], -1))[:min_length]
            set_3d = set_3d[:min_length]
            if config.arch.n_joints == 20 and config.trainer.data == 'cpn':
                set_2d = set_2d.reshape((set_2d.shape[0], -1, 2))
                set_2d = set_2d[:, [0, 1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 8, 11, 12, 13, 8, 14, 15, 16], :]
                set_2d = set_2d.reshape((set_2d.shape[0], -1))

        x = np.tile(set_3d[:, :3], [1, int(set_3d.shape[-1] / 3)])
        set_3d_root = set_3d - np.tile(set_3d[:, :3], [1, int(set_3d.shape[-1] / 3)])
        set_2d_root = set_2d - np.tile(set_2d[:, :2], [1, int(set_2d.shape[-1] / 2)])

        set_2d_root[:, list(range(0, set_2d.shape[-1], 2))] /= res_w
        set_2d_root[:, list(range(1, set_2d.shape[-1], 2))] /= res_h

        set_bones = self.get_bones(set_3d_root)
        set_alphas = np.mean(set_bones, axis=1)
        x = np.expand_dims(set_alphas, axis=-1)

        poses_3d = set_3d_root / np.expand_dims(set_alphas, axis=-1)
        poses_2d = set_2d_root
        if config.trainer.data == 'learnable_with_conf':
            poses_2d = poses_2d.reshape((-1, 20, 2))
            set_2d_conf = set_2d_conf.reshape((-1, 20, 1))
            poses_2d = np.concatenate([poses_2d, set_2d_conf], axis=-1).reshape((-1, 20 * 3))
        # poses_3d_world = h36m_utils.camera_to_world_frame(poses_3d.reshape((-1, 3)), R, T=None).reshape(poses_3d.shape)
        poses_3d_world = set_3d_world
        bones = set_bones / np.expand_dims(set_alphas, axis=-1)
        alphas = set_alphas
        contacts = self.get_contacts(set_3d_world)
        proj_facters = (set_3d / np.expand_dims(set_alphas, axis=-1)).reshape((set_3d.shape[0], -1, 3))[:, 0, 2]

        return x, poses_3d, poses_3d_world, poses_2d, bones, alphas, contacts, proj_facters, set_2d

    def set_sequences(self):
        def slice_set(offset, frame_number, frame_numbers):
            sequence_index = []
            start_index = 0
            for frames in frame_numbers:
                if frames > train_frames:
                    if not self.eval_mode:
                        clips_number = int((frames - train_frames) // offset)
                        for i in range(clips_number):
                            start = int(i * offset + start_index)
                            end = int(i * offset + train_frames + start_index)
                            sequence_index.append(list(range(start, end)))
                        sequence_index.append(list(range(start_index + frames - train_frames, start_index + frames)))
                    else:
                        sequence_index.append(list(range(start_index, start_index + frames)))
                start_index += frames
            return sequence_index

        offset = 10
        train_frames = random.randint(10, 50) * 4 if self.config.trainer.train_frames == 0 else self.config.trainer.train_frames
        train_frames = 196
        self.sequence_index = np.array(slice_set(offset, train_frames, self.frame_numbers))
        self.r_sequence_index = np.array(slice_set(offset, train_frames,  self.r_frame_numbers)) if self.is_train else 0

    def get_flipping(self, poses, n_joints, dim):
        key_left = [4, 5, 6, 11, 12, 13]
        key_right = [1, 2, 3, 14, 15, 16]
        original_shape = poses.shape
        poses_reshape = poses.reshape((-1, n_joints, dim))
        poses_reshape[:, :, 0] *= -1
        poses_reshape[:, key_left + key_right] = poses_reshape[:, key_right + key_left]
        poses_reshape = poses_reshape.reshape(original_shape)
        return poses_reshape

    def get_contacts(self, poses):
        poses_reshape = poses.reshape((-1, self.config.arch.n_joints, 3))
        contact_signal = np.zeros((poses_reshape.shape[0], 2))
        left_z = poses_reshape[:, 3, 2]
        right_z = poses_reshape[:, 6, 2]

        contact_signal[left_z <= (np.mean(np.sort(left_z)[:left_z.shape[0] // 5]) + 20), 0] = 1
        contact_signal[right_z <= (np.mean(np.sort(right_z)[:right_z.shape[0] // 5]) + 20), 1] = 1
        left_velocity = np.sqrt(np.sum((poses_reshape[2:, 3] - poses_reshape[:-2, 3]) ** 2, axis=-1))
        right_velocity = np.sqrt(np.sum((poses_reshape[2:, 6] - poses_reshape[:-2, 6]) ** 2, axis=-1))
        contact_signal[1:-1][left_velocity >= 5, 0] = 0
        contact_signal[1:-1][right_velocity >= 5, 1] = 0
        return contact_signal

    def get_bones(self, position_3d):
        def distance(position1, position2):
            return np.sqrt(np.sum(np.square(position1 - position2), axis=-1))

        length = np.zeros((position_3d.shape[0], 10))

        if self.config.arch.n_joints == 20:
            length[:, 0] = ((distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 1:3 * 1 + 3]) + distance(
                position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 4:3 * 4 + 3])) / 2)
            length[:, 1] = ((distance(position_3d[:, 3 * 1:3 * 1 + 3], position_3d[:, 3 * 2:3 * 2 + 3]) + distance(
                position_3d[:, 3 * 4:3 * 4 + 3], position_3d[:, 3 * 5:3 * 5 + 3])) / 2)
            length[:, 2] = ((distance(position_3d[:, 3 * 2:3 * 2 + 3], position_3d[:, 3 * 3:3 * 3 + 3]) + distance(
                position_3d[:, 3 * 5:3 * 5 + 3], position_3d[:, 3 * 6:3 * 6 + 3])) / 2)
            length[:, 3] = (distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 8:3 * 8 + 3]))
            length[:, 4] = (distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 9:3 * 9 + 3]))
            length[:, 5] = (distance(position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 10:3 * 10 + 3]))
            length[:, 6] = (distance(position_3d[:, 3 * 10:3 * 10 + 3], position_3d[:, 3 * 11:3 * 11 + 3]))
            length[:, 7] = ((distance(position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 13:3 * 13 + 3]) + distance(
                position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 17:3 * 17 + 3])) / 2)
            length[:, 8] = ((distance(position_3d[:, 3 * 17:3 * 17 + 3], position_3d[:, 3 * 18:3 * 18 + 3]) + distance(
                position_3d[:, 3 * 13:3 * 13 + 3], position_3d[:, 3 * 14:3 * 14 + 3])) / 2)
            length[:, 9] = ((distance(position_3d[:, 3 * 18:3 * 18 + 3], position_3d[:, 3 * 19:3 * 19 + 3]) + distance(
                position_3d[:, 3 * 14:3 * 14 + 3], position_3d[:, 3 * 15:3 * 15 + 3])) / 2)
        else:
            length = np.zeros((position_3d.shape[0], 10))
            length[:, 0] = ((distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 1:3 * 1 + 3]) + distance(
                position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 4:3 * 4 + 3])) / 2)
            length[:, 1] = ((distance(position_3d[:, 3 * 1:3 * 1 + 3], position_3d[:, 3 * 2:3 * 2 + 3]) + distance(
                position_3d[:, 3 * 4:3 * 4 + 3], position_3d[:, 3 * 5:3 * 5 + 3])) / 2)
            length[:, 2] = ((distance(position_3d[:, 3 * 2:3 * 2 + 3], position_3d[:, 3 * 3:3 * 3 + 3]) + distance(
                position_3d[:, 3 * 5:3 * 5 + 3], position_3d[:, 3 * 6:3 * 6 + 3])) / 2)
            length[:, 3] = (distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 7:3 * 7 + 3]))
            length[:, 4] = (distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 8:3 * 8 + 3]))
            length[:, 5] = (distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 9:3 * 9 + 3]))
            length[:, 6] = (distance(position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 10:3 * 10 + 3]))
            length[:, 7] = ((distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 11:3 * 11 + 3]) + distance(
                position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 14:3 * 14 + 3])) / 2)
            length[:, 8] = ((distance(position_3d[:, 3 * 14:3 * 14 + 3], position_3d[:, 3 * 15:3 * 15 + 3]) + distance(
                position_3d[:, 3 * 11:3 * 11 + 3], position_3d[:, 3 * 12:3 * 12 + 3])) / 2)
            length[:, 9] = ((distance(position_3d[:, 3 * 15:3 * 15 + 3], position_3d[:, 3 * 16:3 * 16 + 3]) + distance(
                position_3d[:, 3 * 12:3 * 12 + 3], position_3d[:, 3 * 13:3 * 13 + 3])) / 2)
        return length

    def get_parameters(self):
        return self.poses_2d_mean, self.poses_2d_std, self.bones_mean, self.bones_std, self.proj_mean, self.proj_std
