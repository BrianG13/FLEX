import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import base_model
from utils.Quaternions import Quaternions


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class fk_layer(base_model):
    def __init__(self, rotation_type):
        super(fk_layer, self).__init__()
        self.rotation_type = rotation_type
        self.cuda_available = torch.cuda.is_available()

    def normalize_vector(self, v):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(
            torch.FloatTensor([1e-8]).cuda())) if self.cuda_available else torch.max(v_mag, torch.autograd.Variable(
            torch.FloatTensor([1e-8])))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v

    def cross_product(self, u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

        return out

    def transforms_multiply(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def transforms_blank(self, rotations):
        diagonal = torch.diag(torch.ones(4))[None, None, :, :].cuda() if self.cuda_available else torch.diag(
            torch.ones(4))[None, None, :, :]
        ts = diagonal.repeat(int(rotations.shape[0]), int(rotations.shape[1]), 1, 1)
        return ts

    def transforms_rotations(self, rotations):
        if self.rotation_type == 'q':
            q_length = torch.sqrt(torch.sum(torch.pow(rotations, 2), dim=-1))
            qw = rotations[..., 0] / q_length
            qx = rotations[..., 1] / q_length
            qy = rotations[..., 2] / q_length
            qz = rotations[..., 3] / q_length
            qw[qw != qw] = 0
            qx[qx != qx] = 0
            qy[qy != qy] = 0
            qz[qz != qz] = 0
            """Unit quaternion based rotation matrix computation"""
            x2 = qx + qx
            y2 = qy + qy
            z2 = qz + qz
            xx = qx * x2
            yy = qy * y2
            wx = qw * x2
            xy = qx * y2
            yz = qy * z2
            wy = qw * y2
            xz = qx * z2
            zz = qz * z2
            wz = qw * z2

            dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)
            dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)
            dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)
            m = torch.stack([dim0, dim1, dim2], dim=-2)
        elif self.rotation_type == '6d':
            rotations_reshape = rotations.view((-1, 6))
            x_raw = rotations_reshape[:, 0:3]  # batch*3
            y_raw = rotations_reshape[:, 3:6]  # batch*3

            x = self.normalize_vector(x_raw)  # batch*3
            z = self.cross_product(x, y_raw)  # batch*3
            z = self.normalize_vector(z)  # batch*3
            y = self.cross_product(z, x)  # batch*3

            x = x.view(-1, 3, 1)
            y = y.view(-1, 3, 1)
            z = z.view(-1, 3, 1)
            m = torch.cat((x, y, z), 2).reshape((rotations.shape[0], rotations.shape[1], 3, 3))  # batch*3*3
        elif self.rotation_type == 'eular':
            rotations[:, 8, :] = 8
            rotations[:, 15, :2] = 0
            rotations[:, 16, 1:] = 0
            rotations[:, 11, :2] = 0
            rotations[:, 12, 1:] = 0
            rotations_reshape = rotations.view((-1, 3))
            batch = rotations_reshape.shape[0]
            c1 = torch.cos(rotations_reshape[:, 0]).view(batch, 1)  # batch*1
            s1 = torch.sin(rotations_reshape[:, 0]).view(batch, 1)  # batch*1
            c2 = torch.cos(rotations_reshape[:, 2]).view(batch, 1)  # batch*1
            s2 = torch.sin(rotations_reshape[:, 2]).view(batch, 1)  # batch*1
            c3 = torch.cos(rotations_reshape[:, 1]).view(batch, 1)  # batch*1
            s3 = torch.sin(rotations_reshape[:, 1]).view(batch, 1)  # batch*1

            row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
            row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
            row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

            m = torch.cat((row1, row2, row3), 1).reshape((rotations.shape[0], rotations.shape[1], 3, 3))  # batch*3*3
        return m

    def transforms_local(self, positions, rotations):
        cuda_available = torch.cuda.is_available()
        transforms = self.transforms_rotations(rotations)
        if positions.is_cuda and (transforms.is_cuda == False):
            transforms = transforms.cuda()
        transforms = torch.cat([transforms, positions[:, :, :, None]], dim=-1)
        zeros = torch.zeros(
            [int(transforms.shape[0]), int(transforms.shape[1]), 1, 3]).cuda() if cuda_available else torch.zeros(
            [int(transforms.shape[0]), int(transforms.shape[1]), 1, 3])
        ones = torch.ones(
            [int(transforms.shape[0]), int(transforms.shape[1]), 1, 1]).cuda() if cuda_available else torch.ones(
            [int(transforms.shape[0]), int(transforms.shape[1]), 1, 1])
        zerosones = torch.cat([zeros, ones], dim=-1)
        transforms = transforms.double()
        zerosones = zerosones.double()

        transforms = torch.cat([transforms, zerosones], dim=-2)
        return transforms

    def transforms_global(self, parents, positions, rotations):
        locals = self.transforms_local(positions, rotations)
        globals = self.transforms_blank(rotations)
        locals = locals.double()
        globals = globals.double()

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
        globals = list(torch.chunk(globals, int(globals.shape[1]), dim=1))
        for i in range(1, positions.shape[1]):
            globals[i] = self.transforms_multiply(globals[parents[i]][:, 0],
                                                  locals[:, i])[:, None, :, :]
        return torch.cat(globals, dim=1)

    def forward(self, parents, positions, rotations):
        positions = self.transforms_global(parents, positions,
                                           rotations)[:, :, :, 3]
        return positions[:, :, :3] / positions[:, :, 3, None]

    def convert_6d_to_quaternions(self, rotations):
        rotations_reshape = rotations.view((-1, 6))
        x_raw = rotations_reshape[:, 0:3]  # batch*3
        y_raw = rotations_reshape[:, 3:6]  # batch*3

        x = self.normalize_vector(x_raw)  # batch*3
        z = self.cross_product(x, y_raw)  # batch*3
        z = self.normalize_vector(z)  # batch*3
        y = self.cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrices = torch.cat((x, y, z), 2).cpu().numpy()

        q = Quaternions.from_transforms(matrices)

        return q.qs

    def convert_eular_to_quaternions(self, rotations):
        rotations_reshape = rotations.view((-1, 3))
        q = Quaternions.from_euler(rotations_reshape)
        return q.qs

        # batch = matrices.shape[0]
        #
        # w = torch.sqrt(1.0 + matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]) / 2.0
        # w = torch.max(w, torch.autograd.Variable(torch.zeros(batch).cuda()) + 1e-8)  # batch
        # w4 = 4.0 * w
        # x = (matrices[:, 2, 1] - matrices[:, 1, 2]) / w4
        # y = (matrices[:, 0, 2] - matrices[:, 2, 0]) / w4
        # z = (matrices[:, 1, 0] - matrices[:, 0, 1]) / w4
        #
        # quats = torch.cat((w.view(batch, 1), x.view(batch, 1), y.view(batch, 1), z.view(batch, 1)), 1)
        #
        # return quats


class pooling_shrink_net(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number):
        super(pooling_shrink_net, self).__init__()
        print('Branch S - Single-View Configuration')
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv1d(in_features, channel, kernel_size=3, stride=2, bias=True)
        self.expand_bn = nn.BatchNorm1d(channel, momentum=0.1)
        self.shrink = nn.Conv1d(channel, out_features, 1)
        self.stage_number = stage_number
        self.out_features = out_features
        layers = []

        for stage_index in range(0, stage_number):  #
            for conv_index in range(len(kernel_size_set)):
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(channel, channel, kernel_size_set[conv_index], stride_set[conv_index], dilation=1,
                                  bias=True),
                        nn.BatchNorm1d(channel, momentum=0.1)
                    )
                )

        self.stage_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for layer in self.stage_layers:
            x = self.drop(self.relu(layer(x)))
        x = F.adaptive_max_pool1d(x, 1)
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)


class rotation_D(base_model):
    def __init__(self, in_features, out_features, channel, joint_numbers):
        super(rotation_D, self).__init__()
        self.local_fc_layers = nn.ModuleList()
        self.joint_numbers = joint_numbers
        self.shrink_frame_number = 24
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(self.joint_numbers, 500, kernel_size=4, stride=1, bias=False)
        self.conv2 = nn.Conv1d(500, self.joint_numbers, kernel_size=1, stride=1, bias=False)

        for i in range(joint_numbers):
            self.local_fc_layers.append(
                nn.Linear(in_features=self.shrink_frame_number, out_features=1)
            )

    # Get input B*T*J*4
    def forward(self, x):
        x = x.reshape((x.shape[0], -1, self.joint_numbers))
        x = torch.transpose(x, 1, 2)

        x = self.relu(self.conv2(self.relu(self.conv1(x))))
        x = F.adaptive_avg_pool1d(x, self.shrink_frame_number)
        layer_output = []
        for i in range(self.joint_numbers):
            layer_output.append(torch.sigmoid(self.local_fc_layers[i](x[:, i, :].clone())))
        return torch.cat(layer_output, -1)
