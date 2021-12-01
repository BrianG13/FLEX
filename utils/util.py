import os
import numpy as np
import torch
from utils.Quaternions import Quaternions
from utils.quaternion import qfix


ROTATION_NUMBERS = {'q': 4, '6d': 6, 'euler': 3}


def euler_to_quaternions(angles, order, n_joints=20):
    cuda_available = torch.cuda.is_available()
    if isinstance(angles,torch.Tensor):
        angles = angles.cpu().numpy()
    quaternion_angles = Quaternions.from_euler(np.radians(angles), order=order, world=False)
    quaternion_angles = torch.from_numpy(quaternion_angles.qs.reshape((quaternion_angles.shape[0], -1, n_joints * 4))).to(torch.device('cuda:0')) if cuda_available else torch.from_numpy(quaternion_angles.qs.reshape((quaternion_angles.shape[0], -1, n_joints * 4)))
    original_shape = quaternion_angles.shape
    quaternion_angles = quaternion_angles.reshape((-1, n_joints, 4))
    quaternion_angles = qfix(quaternion_angles.cpu().numpy())
    quaternion_angles = quaternion_angles.reshape((-1, original_shape[0], n_joints, 4))
    END_EFFECTORS_INDEXES = [3, 6, 11, 15, 19]
    quaternion_angles[:, :, END_EFFECTORS_INDEXES, :] = 0
    quaternion_angles = quaternion_angles.reshape(original_shape)
    quaternion_angles = torch.from_numpy(quaternion_angles).to(torch.device('cuda:0')) if cuda_available else torch.from_numpy(quaternion_angles)
    return quaternion_angles


def createDict(*args):
    return dict(((k, eval(k)) for k in args))

def mkdir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def make_dataset(dir_list, phase, data_split=3, sort=False, sort_index=1):
    images = []
    for dataroot in dir_list:
        _images = []
        image_filter = []

        assert os.path.isdir(dataroot), '%s is not a valid directory' % dataroot
        for root, _, fnames in sorted(os.walk(dataroot)):
            for fname in fnames:
                if phase in fname:
                    path = os.path.join(root, fname)
                    _images.append(path)
        if sort:
            _images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[sort_index]))
        if data_split is not None:
            for i in range(int(len(_images)/data_split - 1)):
                image_filter.append(_images[data_split*i])
            images += image_filter
            return images
        else:
            return _images

def mkdir(folder):
    if os.path.exists(folder):
        return 1
    else:
        os.makedirs(folder)


def normalize_data(orig_data):
    data_mean = np.mean(orig_data, axis=0)
    data_std = np.std(orig_data, axis=0)
    normalized_data = np.divide((orig_data - data_mean), data_std)
    normalized_data[normalized_data != normalized_data] = 0
    return normalized_data, data_mean, data_std


def umnormalize_data(normalized_data, data_mean, data_std):
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # Dimensionality

    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(normalized_data, stdMat) + meanMat
    return orig_data