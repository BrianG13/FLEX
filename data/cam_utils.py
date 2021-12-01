import numpy as np
from utils import h36m_utils, util
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation as R


def plot2DPose(poses_2d, img_path=None, save_path=None, show=False):
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

    if show:
        plt.show()

def from_R_T_to_extrinsic(R_mat,T_mat):
    temp = np.concatenate([R_mat.T, T_mat], axis=1)
    temp = np.concatenate([temp, np.array([[0, 0, 0, 1]])], axis=0)
    return temp

def quatWAvgMarkley(Q, weights):
    '''
    Averaging Quaternions.

    Arguments:
        Q(ndarray): an Mx4 ndarray of quaternions.
        weights(list): an M elements list, a weight for each quaternion.
    '''

    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    wSum = 0

    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        A += w_i * (np.outer(q, q)) # rank 1 update
        wSum += w_i

    # scale
    A /= wSum

    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(A)[1][:, -1]


def create_cameras_between(cam_a, cam_b, base_name):
    cameras,camera_names = [], []
    R_a, T_a, f, c, k, p, res_w, res_h = cam_a
    R_b, T_b = cam_b[0], cam_b[1]
    r = R.from_matrix(R_a.T)
    r2 = R.from_matrix(R_b.T)
    for alpha in range(0, 10+1, 1):
        alpha /= 10
        t_weighted = alpha * T_a + (1 - alpha) * T_b

        q_weighted = quatWAvgMarkley(np.stack([r.as_quat(), r2.as_quat()]),
                                              [alpha, 1 - alpha])
        r_weighted = R.from_quat(q_weighted).as_matrix().T
        cameras.append([r_weighted, t_weighted, f, c, k, p, res_w, res_h])
        camera_names.append(f"{base_name}_{alpha}")
    return cameras, camera_names

def create_2d_poses_for_synthetic_camera(set_3d_world, cameras):
    import os
    R, T, f, c, k, p, res_w, res_h = cameras[0]
    R_b, T_b, f, c, k, p, res_w, res_h = cameras[1]
    cameras, cameras_names = create_cameras_between(cameras[0], cameras[1])
    poses_2d = []
    i = 0.1
    for cam in cameras:
        r_mat, t_mat = cam[0], cam[1]
        set_2d = h36m_utils.project_2d(set_3d_world.reshape((-1, 3)), r_mat, t_mat, f, c, k, p, from_world=True)[
            0].reshape((set_3d_world.shape[0], int(set_3d_world.shape[-1] / 3 * 2)))
        poses_2d.append(set_2d)
        BASE_DATA_PATH = os.path.dirname(os.path.realpath(__file__))
        plot2DPose(set_2d[0],save_path=f"{BASE_DATA_PATH}/test_2d_{i:.1f}.png")
        i+=0.1
