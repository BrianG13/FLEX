import torch
import numpy as np
from utils.Quaternions import Quaternions
from utils.quaternion import qfix, qeuler
import utils.quaternion as quaternion
from model import model

def mean_angle_error(fake_rotations_full, quaternion_angles):
    with torch.no_grad():
        fake_angles = fake_rotations_full.reshape((-1, 4))
        norm = torch.norm(fake_angles, dim=-1)
        norm[norm == 0] = 1
        fake_angles = fake_angles / norm.reshape((-1, 1))
        fake_angles = np.degrees(qeuler(fake_angles, order='zxy').cpu().numpy())
        quaternion_angles = quaternion_angles.reshape((-1, 4))
        gt_angles = np.degrees(qeuler(quaternion_angles, order='zxy').cpu().numpy())
        diff = fake_angles - gt_angles
        sign = np.sign(diff)
        sign[sign == 0] = 1
        error = np.mean(np.abs(np.mod(diff, sign * 180)))
    return error


def mean_angle_error_pavllo(fake_rotations_full, quaternion_angles, n_joints=20):
    with torch.no_grad():
        predicted_quat = fake_rotations_full.reshape((-1, 4))
        norm = torch.norm(predicted_quat, dim=-1)
        norm[norm == 0] = 1
        predicted_quat = predicted_quat / norm.reshape((-1, 1))
        predicted_quat = predicted_quat.view(-1, n_joints, 4)

        predicted_euler = qeuler(predicted_quat, order='zxy', epsilon=1e-6)

        expected_quat = quaternion_angles.view(-1, n_joints, 4)
        expected_euler = qeuler(expected_quat, order='zxy', epsilon=1e-6)

        # L1 loss on angle distance with 2pi wrap-around
        angle_distance = torch.remainder(predicted_euler - expected_euler + np.pi, 2 * np.pi) - np.pi
        return torch.mean(torch.abs(angle_distance))


def mean_points_error(output, target):
    with torch.no_grad():
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
        if not isinstance(target, np.ndarray):
            target = target.data.cpu().numpy()
        output_reshape = output.reshape((-1, 3))
        target_reshape = target.reshape((-1, 3))
        error = np.mean(np.sqrt(np.sum(np.square((output_reshape - target_reshape)), axis=1)))
    return error


def compute_error_accel(joints_gt, joints_pred,alphas=None, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    joints_gt = np.squeeze(joints_gt.detach().cpu().numpy())
    joints_pred = np.squeeze(joints_pred.detach().cpu().numpy())
    if alphas is not None:
        alphas = np.squeeze(alphas.detach().cpu().numpy())
        joints_gt = joints_gt * np.expand_dims(alphas, axis=-1)
        joints_pred = joints_pred * np.expand_dims(alphas, axis=-1)

    joints_gt = joints_gt.reshape((joints_gt.shape[0], -1, 3))
    joints_pred = joints_pred.reshape((joints_pred.shape[0], -1, 3))

    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(np.mean(normed[new_vis], axis=1))


def mean_points_error_per_joint(output, target, n_joints):
    joint_errors = []
    with torch.no_grad():
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()
        output_reshape = output.reshape((-1, n_joints, 3))
        target_reshape = target.reshape((-1, n_joints, 3))
        for i in range(n_joints):
            output_joints = output_reshape[:, i, :]
            target_joints = target_reshape[:, i, :]
            error = np.mean(np.sqrt(np.sum(np.square((output_joints - target_joints)), axis=1)))
            joint_errors.append(error)
    return joint_errors


def mean_points_error_index(output, target):
    with torch.no_grad():
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()
        output_reshape = output.reshape((-1, 3))
        target_reshape = target.reshape((-1, 3))
        error_vector = np.sqrt(np.sum(np.square((output_reshape - target_reshape)), axis=1))
    return np.argmax(error_vector)


def mean_points_error_sequence(output, target):
    with torch.no_grad():
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()
        output_reshape = output.reshape((-1, 3))
        target_reshape = target.reshape((-1, 3))
        # if len(output.shape) == 3:
        #     # output_reshape = output.reshape((output.shape[0] * int(output.shape[1]/2)*output.shape[2], 2))
        #     output_reshape = output.reshape((-1, 3))
        #     # target_reshape = target.reshape((output.shape[0] * int(output.shape[1]/2)*output.shape[2], 2))
        #     target_reshape = target.reshape((-1, 3))
        # elif len(output.shape) == 2:
        #     # output_reshape = output.reshape((output.shape[0] * int(output.shape[1] / 2), 2))
        #     # target_reshape = target.reshape((output.shape[0] * int(output.shape[1] / 2), 2))
        error = np.mean(np.sqrt(np.sum(np.square((output_reshape - target_reshape)), axis=1)))
    return error
