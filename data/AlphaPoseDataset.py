import json
# -*- coding:utf-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
from utils import h36m_utils, util
from pathlib import Path


class AlphaPoseDataset(Dataset):
    def __init__(self, config, is_train=False, num_of_views=3):
        print('AlphaPoseDataset')

        poses_3d, poses_2d = [], []

        self.frame_numbers = []
        self.video_and_frame = []
        self.video_name = []
        self.config = config
        self.is_train = is_train

        self.n_camera_views = 4

        for ACTION_NAME in ['Fight_Scene']:
            # for cameras_idx in itertools.permutations([10,11,12,13],4):
            current_dir = os.path.abspath(os.getcwd())
            path = Path(__file__).parent.absolute()

            with open(os.path.join(path, f"alphapose/{ACTION_NAME}_alphapose.json"), "r") as read_file:
                action_json = json.load(read_file)

            for subject in action_json:
                subject_data = action_json[subject]
                for cameras_idx in [[18, 19, 20, 21], [18, 19, 21, 22], [19, 20, 22, 23]]:
                    # for cameras_idx in [[12, 11, 10, 13]]:
                    set_views_poses_2d = []
                    min_length = float('inf')
                    for cam_idx in cameras_idx:
                        subject_cam_data = subject_data[str(cam_idx)]
                        set_2d = self.subject_cam_data_to_pose2d(subject_cam_data)
                        set_2d = set_2d.reshape((-1, 20 * 2))
                        if set_2d.shape[0] < min_length:
                            min_length = set_2d.shape[0]
                        set_2d = self.center_and_scale(set_2d, 800, 800)
                        set_views_poses_2d.append(set_2d)

                    self.frame_numbers.append(min_length)
                    string_idxs = [str(n) for n in cameras_idx]
                    self.video_name.append(f"{ACTION_NAME}_S{subject}_" + "_".join(string_idxs))
                    for i in range(len(set_views_poses_2d)):
                        set_views_poses_2d[i] = set_views_poses_2d[i][:min_length, :]
                    poses_2d.append(np.stack(set_views_poses_2d, axis=0))
        self.poses_2d = np.concatenate(poses_2d, axis=1)
        self.set_sequences()

        self.poses_2d = self.poses_2d.reshape((-1, self.poses_2d.shape[-1]))

        print('Normalizing Poses 2D ..')
        self.poses_2d, self.poses_2d_mean, self.poses_2d_std = util.normalize_data(self.poses_2d)
        self.poses_2d = self.poses_2d.reshape((self.n_camera_views, -1, self.poses_2d.shape[-1]))
        self.translate_to_lists()

    def subject_cam_data_to_pose2d(self, subject_cam_data):
        n_frames = len(subject_cam_data)
        all_keypoints = []
        for frame_data in subject_cam_data:
            all_keypoints.append(frame_data['keypoints'])
        all_keypoints = np.asarray(all_keypoints).reshape((n_frames, 20, 3))

        all_keypoints = all_keypoints[:, :, :2]  # Remove confidence
        all_keypoints[:, 8, :] = 0.5 * (all_keypoints[:, 8, :2] + all_keypoints[:, 9, :2])
        return all_keypoints

    def translate_to_lists(self):
        # self.poses_3d_list = []
        self.poses_2d_list = []

        for i in range(self.n_camera_views):
            # self.poses_3d_list.append(self.poses_3d[i])
            self.poses_2d_list.append(self.poses_2d[i])

    def __getitem__(self, index):
        items_index = self.sequence_index[index]
        # Start of experiment"
        views_data = []
        for i in range(self.n_camera_views):
            # poses_3d = self.poses_3d_list[i][items_index]
            poses_2d = self.poses_2d_list[i][items_index]

            # views_data.append([poses_2d, poses_3d])
            views_data.append([poses_2d])

        return views_data, self.video_name[index]

    def __len__(self):
        return self.sequence_index.shape[0]

    def get_video_and_frame_details(self, index):
        items_index = self.sequence_index[index]
        return self.video_and_frame[items_index]

    def center_and_scale(self, set_2d, res_height=640, res_width=480):
        set_2d_root = set_2d - np.tile(set_2d[:, :2], [1, int(set_2d.shape[-1] / 2)])
        set_2d_root[:, list(range(0, set_2d.shape[-1], 2))] /= res_height
        set_2d_root[:, list(range(1, set_2d.shape[-1], 2))] /= res_width
        return set_2d_root

    def set_sequences(self):
        def slice_set(offset, frame_number, frame_numbers):
            sequence_index = []
            start_index = 0
            for frames in frame_numbers:
                if frames > train_frames:
                    if self.is_train:
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
        train_frames = 0
        self.sequence_index = np.array(slice_set(offset, train_frames, self.frame_numbers))

    def get_parameters(self):
        bones_mean = np.asarray(
            [[0.516709056080739, 1.8597128599473656, 1.8486176161353194, 1.0154634542775278, 0.9944090324875341,
              0.46335894047448684, 0.45708159715181057, 0.7002218481026042, 1.155838079375657,
              0.9885875159669523]])
        bones_std = np.asarray([0.03250722274696495, 0.0029220969699794838, 0.0006156061152456668, 0.005663153869630782,
                                0.008404610836805063, 0.014311367462960448, 0.004236793633603145, 0.026840327186035892,
                                0.01512937262086334, 0.007775925244763229])

        return self.poses_2d_mean, self.poses_2d_std, bones_mean, bones_std


#
if __name__ == '__main__':
    dataset = AlphaPoseDataset(config={'run_local': True}, num_of_views=3)
