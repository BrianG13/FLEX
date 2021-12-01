import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import base_model
import numpy as np
from .transformer_model import TransformerModel

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class pooling_shrink_net(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number, n_views):
        super(pooling_shrink_net, self).__init__()
        print('Branch S - Multi-View Configuration')
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv2d(in_features, channel, kernel_size=(3,1), stride=1, bias=True)
        self.expand_bn = nn.BatchNorm2d(channel, momentum=0.1)
        self.shrink = nn.Conv1d(channel, out_features, 1)
        self.stage_number = stage_number
        self.out_features = out_features
        self.n_views = n_views
        self.fusion = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=(1, self.n_views), stride=1, dilation=1, bias=True),nn.BatchNorm2d(channel, momentum=0.1))
        layers = []

        for stage_index in range(0, stage_number):  #
            for conv_index in range(len(kernel_size_set)):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(channel, channel, (kernel_size_set[conv_index], 1), stride_set[conv_index],
                                  dilation=1, bias=True),
                        nn.BatchNorm2d(channel, momentum=0.1)
                    )
                )

        self.stage_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for layer in self.stage_layers:
            x = self.drop(self.relu(layer(x)))
        x = self.relu(self.fusion(x))
        x = x.squeeze(dim=-1)
        x = F.adaptive_max_pool1d(x, 1)
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)


class pooling_net_late_fusion(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number,
                 kernel_size_set_stage_1, kernel_size_set_stage_2, config=None):
        super(pooling_net_late_fusion, self).__init__()
        self.config = config
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.expand_conv = nn.Conv2d(in_features, channel, kernel_size=(1,3),padding=(0,1),padding_mode='reflect', stride=1, bias=True)
        self.expand_conv = nn.Conv2d(in_features, channel, kernel_size=(1, 1), stride=1, bias=True)
        self.expand_bn = nn.BatchNorm2d(channel, momentum=0.1)
        self.stage_number = stage_number
        self.conv_depth = len(kernel_size_set)
        self.out_features = out_features
        self.kernel_size_set_stage_1 = kernel_size_set_stage_1 if kernel_size_set_stage_1 else kernel_size_set
        self.kernel_size_set_stage_2 = kernel_size_set_stage_2 if kernel_size_set_stage_2 else kernel_size_set
        self.fusion = nn.Sequential(nn.Conv2d(channel, channel, (1, config.arch.n_views), 1, dilation=1, bias=True),nn.BatchNorm2d(channel, momentum=0.1))
        self.shrink = nn.Conv1d(channel, out_features, kernel_size=1, stride=1, bias=True)
        self.padding_size = self.config.arch.padding if getattr(self.config.arch, 'padding', False) else 10
        self.kernel_width = self.config.arch.kernel_width if getattr(self.config.arch, 'kernel_width', False) else 100
        print('LATE Fusion Network')
        print(f'Number of views: {config.arch.n_views}')
        print(f'Kernel width: {self.kernel_width}')
        print(f'Padding size: {self.padding_size}')
        PADDING_MODE = 'circular'
        layers = []
        # 1st stage convolutions
        for conv_index in range(len(self.kernel_size_set_stage_1)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=(kernel_size_set_stage_1[conv_index], 1),
                              stride=stride_set[conv_index], dilation=1, bias=True),
                    nn.BatchNorm2d(channel, momentum=0.1)
                )
            )
        # Fusion layer
        # 2nd stage convolutions
        for conv_index in range(len(kernel_size_set_stage_2)):
            seq = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=(kernel_size_set_stage_2[conv_index], 1),
                          stride=stride_set[conv_index], dilation=1,
                          padding=(0, self.padding_size), padding_mode='reflect', bias=True),
                nn.BatchNorm2d(channel, momentum=0.1))
            layers.append(seq)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # [BatchSize, #Views, #Frames, #Joints]
        x = torch.transpose(x, 1, 3)
        # [BatchSize, #Joints, #Frames, #Views]

        x = self.expand_conv(x)
        x = self.drop(self.relu(self.expand_bn(x)))

        # 1st stage forward
        output = 0
        for conv_index in range(len(self.kernel_size_set_stage_1)):
            temp = F.adaptive_avg_pool2d(self.drop(self.relu(self.layers[conv_index](x))), (x.shape[-2],x.shape[-1]))
            output += temp
        x = output
        # 2nd stage forward
        output = 0
        for conv_index in range(len(self.kernel_size_set_stage_1),len(self.kernel_size_set_stage_1)+len(self.kernel_size_set_stage_2)):
            layer = self.layers[conv_index]
            temp = F.adaptive_avg_pool2d(self.drop(self.relu(layer(x))), (x.shape[-2],x.shape[-1]))
            output += temp
        x = output
        x = self.fusion(x)
        x = x.squeeze(dim=-1)
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)


class pooling_net(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number,
                 kernel_size_set_stage_1, kernel_size_set_stage_2, config=None):
        super(pooling_net, self).__init__()
        self.config = config
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.expand_conv = nn.Conv2d(in_features, channel, kernel_size=(1,3),padding=(0,1),padding_mode='reflect', stride=1, bias=True)
        self.expand_conv = nn.Conv2d(in_features, channel, kernel_size=(1,1), stride=1, bias=True)
        self.expand_bn = nn.BatchNorm2d(channel, momentum=0.1)
        self.stage_number = stage_number
        self.conv_depth = len(kernel_size_set)
        self.out_features = out_features
        self.kernel_size_set_stage_1 = kernel_size_set_stage_1 if kernel_size_set_stage_1 else kernel_size_set
        self.kernel_size_set_stage_2 = kernel_size_set_stage_2 if kernel_size_set_stage_2 else kernel_size_set
        self.fusion = nn.Sequential(nn.Conv2d(channel, channel, (1, config.arch.n_views), 1, dilation=1, bias=True),nn.BatchNorm2d(channel, momentum=0.1))
        self.shrink = nn.Conv1d(channel, out_features, kernel_size=1, stride=1, bias=True)
        self.padding_size = self.config.arch.padding if getattr(self.config.arch, 'padding', False) else 10
        self.kernel_width = self.config.arch.kernel_width if getattr(self.config.arch, 'kernel_width', False) else 100
        print('Middle Fusion Network')
        print(f'Number of views: {config.arch.n_views}')
        print(f'Kernel width: {self.kernel_width}')
        print(f'Padding size: {self.padding_size}')
        PADDING_MODE = 'circular'
        layers = []
        # 1st stage convolutions
        for conv_index in range(len(self.kernel_size_set_stage_1)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=(kernel_size_set_stage_1[conv_index], 1),
                              stride=stride_set[conv_index], dilation=1, bias=True),
                    nn.BatchNorm2d(channel, momentum=0.1)
                )
            )
        # Fusion layer
        # 2nd stage convolutions
        for conv_index in range(len(kernel_size_set_stage_2)):
            seq = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=(kernel_size_set_stage_2[conv_index], self.kernel_width),
                          stride=stride_set[conv_index], dilation=1,
                          padding=(0, self.padding_size), padding_mode='reflect', bias=True),
                nn.BatchNorm2d(channel, momentum=0.1))
            layers.append(seq)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # [BatchSize, #Views, #Frames, #Joints]
        x = torch.transpose(x, 1, 3)
        # [BatchSize, #Joints, #Frames, #Views]

        x = self.expand_conv(x)
        x = self.drop(self.relu(self.expand_bn(x)))

        # 1st stage forward
        output = 0
        for conv_index in range(len(self.kernel_size_set_stage_1)):
            temp = F.adaptive_avg_pool2d(self.drop(self.relu(self.layers[conv_index](x))), (x.shape[-2],x.shape[-1]))
            output += temp
        x = output
        # 2nd stage forward
        output = 0
        for conv_index in range(len(self.kernel_size_set_stage_1),len(self.kernel_size_set_stage_1)+len(self.kernel_size_set_stage_2)):
            layer = self.layers[conv_index]
            temp = F.adaptive_avg_pool2d(self.drop(self.relu(layer(x))), (x.shape[-2],x.shape[-1]))
            output += temp
        x = output
        x = self.fusion(x)
        x = x.squeeze(dim=-1)
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)

class pooling_net_early_fusion(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number,
                 kernel_size_set_stage_1, kernel_size_set_stage_2, config=None):
        super(pooling_net_early_fusion, self).__init__()
        self.config = config
        self.gpu0_device = torch.device("cuda:0")
        self.gpu1_device = torch.device("cuda:1")
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv2d(in_features, channel, kernel_size=(1,1), stride=1, bias=True)
        self.expand_bn = nn.BatchNorm2d(channel, momentum=0.1)
        self.stage_number = stage_number
        self.conv_depth = len(kernel_size_set)
        self.out_features = out_features
        self.kernel_size_set_stage_1 = kernel_size_set_stage_1 if kernel_size_set_stage_1 else kernel_size_set
        self.kernel_size_set_stage_2 = kernel_size_set_stage_2 if kernel_size_set_stage_2 else kernel_size_set
        if config.arch.n_views == 1:
            self.expand_conv = nn.Conv1d(in_features, channel, kernel_size=1, stride=1, bias=True)
            self.expand_bn = nn.BatchNorm1d(channel, momentum=0.1)
        else:
            self.expand_conv = nn.Conv2d(in_features, channel, kernel_size=(1, 1), stride=1, bias=True)
            self.expand_bn = nn.BatchNorm2d(channel, momentum=0.1)

        PADDING_MODE = 'circular'
        if self.config.arch.padding == 0:
            self.padding_size = 0
        else:
            self.padding_size = self.config.arch.padding if getattr(self.config.arch, 'padding', False) else 10
        self.kernel_width = self.config.arch.kernel_width if getattr(self.config.arch, 'kernel_width', False) else 100
        print('Early Fusion Network')
        print(f'Number of views: {config.arch.n_views}')
        print(f'Kernel width: {self.kernel_width}')
        print(f'Padding size: {self.padding_size}')
        print(f'Kernels set Stage 1: {self.kernel_size_set_stage_1}')
        print(f'Kernels set Stage 2: {self.kernel_size_set_stage_2}')


        if getattr(self.config.arch, 'transformer_on', False):
            self.transformerModel = TransformerModel(config=config,
                                                     input_dim=channel, output_dim=channel,
                                                     n_heads=self.config.arch.transformer_n_heads,
                                                     n_layers=self.config.arch.transformer_n_layers)
        elif config.arch.n_views != 1:
            self.fusion = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=(1, config.arch.n_views), stride=1, dilation=1, bias=True),
                nn.BatchNorm2d(channel, momentum=0.1))

        layers = []
        # 1st stage convolutions
        for conv_index in range(len(self.kernel_size_set_stage_1)):
            if config.arch.n_views == 1:
                layer = nn.Sequential(
                        nn.Conv1d(channel, channel, kernel_size=kernel_size_set_stage_1[conv_index],
                                stride=stride_set[conv_index], dilation=1, bias=True),
                        nn.BatchNorm1d(channel, momentum=0.1)
                    )
            else:
                layer = nn.Sequential(
                        nn.Conv2d(channel, channel, kernel_size=(kernel_size_set_stage_1[conv_index], self.kernel_width),
                                stride=stride_set[conv_index], dilation=1,
                                padding=(0, self.padding_size), padding_mode=PADDING_MODE, bias=True),
                        nn.BatchNorm2d(channel, momentum=0.1)
                    )
            layers.append(layer)
        # Fusion layer
        # 2nd stage convolutions
        for conv_index in range(len(self.kernel_size_set_stage_2)):
            seq = nn.Sequential(
                nn.Conv1d(channel, channel, kernel_size=kernel_size_set_stage_2[conv_index],
                          stride=stride_set[conv_index], dilation=1, bias=True),
                nn.BatchNorm1d(channel, momentum=0.1)
                )
            layers.append(seq)

        self.shrink = nn.Conv1d(channel, out_features, kernel_size=1, stride=1, bias=True)
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        # [BatchSize, #Views, #Frames, #Joints]
        x = torch.transpose(x, 1, 3)
        # [BatchSize, #Joints, #Frames, #Views]
        if self.config.arch.n_views == 1:
            x = x.squeeze(dim=-1)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        # 1st stage forward
        output = 0
        for conv_index in range(len(self.kernel_size_set_stage_1)):
            if self.config.arch.n_views == 1:
                temp = F.adaptive_avg_pool1d(self.drop(self.relu(self.layers[conv_index](x))), x.shape[-1])
            else:
                temp = self.drop(self.relu(self.layers[conv_index](x)))
                temp = F.adaptive_avg_pool2d(temp, (x.shape[-2], x.shape[-1]))
            output += temp
        x = output
        # Fusion to narrow tensor:
        if self.config.arch.n_views != 1:
            if getattr(self.config.arch, 'transformer_on', False):
                x = self.transformerModel(x)
            else:
                x = self.drop(self.relu(self.fusion(x)))
            x = x.squeeze(dim=-1)
        # 2nd stage forward

        if len(self.kernel_size_set_stage_2) > 0:
            output = 0
            for conv_index in range(len(self.kernel_size_set_stage_1),
                                    len(self.kernel_size_set_stage_1) + len(self.kernel_size_set_stage_2)):
                layer = self.layers[conv_index]
                temp = F.adaptive_avg_pool1d(self.drop(self.relu(layer(x))), x.shape[-1])
                output += temp
            x = output
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)
