
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import cv2
import numpy as np
from cam_utils import from_R_T_to_extrinsic, quatWAvgMarkley

class CameraPoseVisualizer:
    def __init__(self, axis=None):
        self.fig = plt.figure(figsize=(18, 7))
        if axis is None:
            self.ax = self.fig.gca(projection='3d')
        else:
            self.ax = axis

        self.ax.set_aspect("auto")
        # self.ax.set_xlim([-5, 5])
        # self.ax.set_ylim([-5, 5])
        # self.ax.set_zlim([-5, 5])
        self.ax.set_xlim([-5000, 5000])
        self.ax.set_ylim([-5000, 5000])
        self.ax.set_zlim([-5000, 5000])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=2, aspect_ratio=0.3,name=None):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        # for i in range(len(meshes)):
        #     for j in range(len(meshes[i])):
        #         meshes[i][j] = meshes[i][j][[0,2,1]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))
        if name:
            T = extrinsic[:3, 3]
            self.ax.text(T[0], T[1], T[2], name)

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

# def from_R_T_to_extrinsic(R_mat,T_mat):
#     temp = np.concatenate([R_mat.T, T_mat], axis=1)
#     temp = np.concatenate([temp, np.array([[0, 0, 0, 1]])], axis=0)
#     return temp


if __name__ == '__main__':
    # import matplotlib
    # matplotlib.rcParams["backend"] = "TkAgg"
    gs1 = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs1[0, 0], projection='3d')
    visualizer = CameraPoseVisualizer(ax)
    R_list = [
        np.asarray([[-0.90334862, 0.42691198, 0.04132109],
                    [0.04153061, 0.18295114, -0.98224441],
                    [-0.42689165, -0.88559305, -0.18299858]]),
        np.asarray([[0.93157205, 0.36348288, -0.00732918],
                    [0.06810069, -0.19426748, -0.97858185],
                    [-0.35712157, 0.91112038, -0.20572759]]),
        np.asarray([[-0.92693442, -0.37323035, -0.03862235],
                    [-0.04725991, 0.21824049, -0.97475001],
                    [0.37223525, -0.90170405, -0.21993346]]),
        np.asarray([[0.91546071, -0.39734607, 0.0636223],
                     [-0.04940628, -0.26789168, -0.96218141],
                     [0.39936288, 0.87769594, -0.2648757]])
    ]
    T_list = [
        np.asarray([[2044.45852504],
                    [4935.11727985],
                    [1481.22752753]]),
        np.asarray([[1990.95966215],
                    [-5123.81055156],
                    [1568.80481574]]),
        np.asarray([[-1670.99215489],
                    [5211.98574196],
                    [1528.38799772]]),
        np.asarray([[-1696.04347097],
                    [-3827.09988629],
                    [1591.41272728]])
    ]
    for cam_idx in range(len(R_list)):
        cam_extrinsics = from_R_T_to_extrinsic(R_list[cam_idx], T_list[cam_idx])
        visualizer.extrinsic2pyramid(cam_extrinsics, 'r', 1000, name=str(cam_idx))

    # from scipy.spatial.transform import Rotation as R
    # r = R.from_matrix(R_list[2].T)
    # r2 = R.from_matrix(R_list[3].T)
    # for alpha in range(1, 10, 1):
    #     alpha /= 10
    #     # r_avg = 0.5*(r.as_rotvec()+r2.as_rotvec())
    #     # r_avg = R.from_rotvec(r_avg).as_matrix()
    #     t_avg = alpha * T_list[2] + (1 - alpha) * T_list[3]
    #     # cam_extrinsics = from_R_T_to_extrinsic(r_avg.T, t_avg)
    #
    #     all_q = np.stack([r.as_quat(), r2.as_quat()])
    #     avg_q = quatWAvgMarkley(all_q, [alpha, 1 - alpha])
    #     avg_r = R.from_quat(avg_q).as_matrix()
    #     cam_extrinsics = from_R_T_to_extrinsic(avg_r.T, t_avg)
    #     visualizer.extrinsic2pyramid(cam_extrinsics, 'c', 1000, name=str(alpha))

    # ax.view_init(azim=-20, elev=20)
    plt.show()
