import os
import copy
import json
import argparse
import torch
import model.model as models
from data.data_loaders import h36m_loader
from trainer.multi_view_trainer import fk_trainer_multi_view
from types import SimpleNamespace as Namespace

base_path = os.path.dirname(os.path.realpath(__file__))


def get_clearml_logger(config):
    import clearml
    import trains
    if config.arch.transformer_on:
        project_name = f'MotioNet-MultiView - {config.arch.n_views} Views - Transformer: {config.arch.transformer_mode}'
        task_name = f'[{config.trainer.data}][{config.arch.transformer_n_heads} Heads][{config.arch.transformer_n_layers} Layers](3D pos.loss)'
    else:
        project_name = f'MotioNet-MultiView - {config.arch.n_views} Views'
        task_name = f'[{config.trainer.data}] (3D pos.loss) Early fusion'

    if config.arch.translation:
        task_name += " [Translation]"

    if config.trainer.optimizer == "adaw":
        task_name += " [AdaW]"

    if config.trainer.train_only_on_cameras is not None:
        task_name += f" Training only on camera tuples: {config.trainer.train_only_on_cameras}"

    if getattr(config, 'momo', False):
        task = trains.Task.init(project_name=project_name, task_name=task_name)
        clearml_logger = task.get_logger()
    else:
        task = clearml.Task.init(project_name=project_name, task_name=task_name)
        clearml_logger = task.get_logger()

    return clearml_logger

def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def config_parse(args):
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))

    config.device = str(args.device)
    config.arch.transformer_on = args.transformer_on if args.transformer_on is not None else config.arch.transformer_on
    config.arch.transformer_n_heads = args.transformer_n_heads
    config.arch.transformer_n_layers = args.transformer_n_layers
    config.arch.transformer_mode = args.transformer_mode

    config.momo = bool(args.momo)
    config.arch.n_joints = int(args.n_joints)
    config.trainer.starting_pos_totem = args.starting_pos_totem
    config.trainer.arms_and_legs_weight = args.arms_and_legs_weight
    config.trainer.use_3d_world_pose_as_labels = args.use_3d_world_pose_as_labels
    config.arch.branch_S_multi_view = args.branch_S_multi_view

    config.arch.kernel_size = list(map(int, args.kernel_size.replace(' ', '').strip().split(','))) if args.kernel_size is not None else config.arch.kernel_size
    config.arch.stride = list(map(int, args.stride.replace(' ', '').strip().split(','))) if args.stride is not None else config.arch.stride
    config.arch.dilation = list(map(int, args.dilation.replace(' ', '').strip().split(','))) if args.dilation is not None else config.arch.dilation

    config.arch.kernel_size_stage_1 = list(map(int, args.kernel_size_stage_1.replace(' ', '').strip().split(','))) if args.kernel_size_stage_1 is not None else config.arch.kernel_size_stage_1
    config.arch.stride_stage_1 = list(map(int, args.stride_stage_1.replace(' ', '').strip().split(','))) if args.stride_stage_1 is not None else config.arch.stride_stage_1
    config.arch.dilation_stage_1 = list(map(int, args.dilation_stage_1.replace(' ', '').strip().split(','))) if args.dilation_stage_1 is not None else config.arch.dilation_stage_1
    config.arch.kernel_size_stage_2 = list(map(int, args.kernel_size_stage_2.replace(' ', '').strip().split(','))) if args.kernel_size_stage_2 is not None else config.arch.kernel_size_stage_2

    config.arch.n_views = int(args.n_views) if args.n_views is not None else config.arch.n_views
    config.arch.kernel_width = int(args.kernel_width) if args.kernel_width is not None else None
    config.arch.padding = int(args.padding) if args.padding is not None else None

    config.arch.channel = int(args.channel) if args.channel is not None else config.arch.channel
    config.arch.stage = int(args.stage) if args.stage is not None else config.arch.stage
    config.arch.n_type = int(args.n_type) if args.n_type is not None else config.arch.n_type
    config.arch.rotation_type = args.rotation_type if args.rotation_type is not None else config.arch.rotation_type
    config.arch.translation = True if args.translation == 1 else config.arch.translation
    config.arch.confidence = True if args.confidence == 1 else config.arch.confidence
    config.arch.contact = True if args.contact == 1 else config.arch.contact
    if args.train_only_on_cameras is not None:
        camera_tuples = list(args.train_only_on_cameras.replace(' ', '').strip().split(':'))
        config.trainer.train_only_on_cameras = []
        for cam_idxs in camera_tuples:
            config.trainer.train_only_on_cameras.append(list(map(int, cam_idxs.split(','))))
    else:
        config.trainer.train_only_on_cameras = None
    config.trainer.data = args.data
    config.trainer.lr = args.lr
    config.trainer.batch_size = args.batch_size
    config.trainer.train_frames = args.train_frames
    config.trainer.use_loss_foot = True if args.loss_terms[0] == '1' else False
    config.trainer.use_loss_3d = True if args.loss_terms[1] == '1' else False
    config.trainer.use_loss_2d = True if args.loss_terms[2] == '1' else False
    config.trainer.use_loss_D = True if args.loss_terms[3] == '1' else False
    config.trainer.data_aug_flip = True if args.augmentation_terms[0] == '1' else False
    config.trainer.data_aug_depth = True if args.augmentation_terms[1] == '1' else False
    config.trainer.save_dir = args.save_dir if args.save_dir is not None else base_path

    config.trainer.checkpoint_dir = '%s/%s_%s_k%s_s%s_d%s_c%s_%s_%s_%s%s%s_%s_%s_loss%s_aug%s' % (config.trainer.save_dir, args.name, args.data, 
                                                                                           str(config.arch.kernel_size).strip('[]').replace(' ', ''),
                                                                                           str(config.arch.stride).strip('[]').replace(' ', ''),
                                                                                           str(config.arch.dilation).strip('[]').replace(' ', ''),
                                                                                           config.arch.channel, config.arch.stage, config.arch.rotation_type, 
                                                                                           't' if config.arch.translation else '',
                                                                                           'c' if config.arch.confidence else '',
                                                                                           'c' if config.arch.contact else '',
                                                                                           args.lr, args.batch_size, args.loss_terms, args.augmentation_terms)
    return config


def train(config, resume):
    print("Loading dataset..")
    train_data_loader = h36m_loader(config, is_training=True)
    test_data_loader = h36m_loader(config, is_training=False)

    model = getattr(models, config.arch.type)(config)
    # print(model.summary())
    count_parameters(model)
    trainer = fk_trainer_multi_view(model, resume=resume, config=config, data_loader=train_data_loader,
                                    test_data_loader=test_data_loader, clearml_logger=clearml_logger,
                                    num_of_views=config.arch.n_views)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='### MotioNet training')

    parser.add_argument('--clearml_logger', action='store_true', help='Turn on ClearML Framework')

    # Runtime parameters
    parser.add_argument('--transformer_on', action='store_true', help='Activate Attention Heads on Fusion layer')
    parser.add_argument('--transformer_n_heads', default=2, type=int, help='Number of heads to use')
    parser.add_argument('--transformer_n_layers', default=2, type=int, help='Number of heads to use')
    parser.add_argument('--transformer_mode', default="encoder", type=str, help='Number of heads to use')

    parser.add_argument('-m', '--momo', action='store_true', help='If we are running on momo server')
    parser.add_argument('--n_joints', default=20, type=int, help='Number of joints to use')
    parser.add_argument('--starting_pos_totem', action='store_true',default=True, help='If we are running on momo server')
    parser.add_argument('--arms_and_legs_weight', type=float, default=1, help='If we are running on momo server')
    parser.add_argument('-c', '--config', default=f'{base_path}/config_zoo/default.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='2', type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--name', default='debug_model', type=str, help='The name of this training')
    parser.add_argument('--data', default='gt', type=str,
                        help='The training data, gt - projected 2d pose; cpn; detectron')
    parser.add_argument('--use_3d_world_pose_as_labels', action='store_true', help='If we are running on momo server')
    parser.add_argument('--branch_S_multi_view', action='store_true', help='If we are running on momo server')


    # Network definition
    parser.add_argument('--kernel_size', default=None, type=str, help='The kernel_size set of the convolution')
    parser.add_argument('--stride', default=None, type=str, help='The stride set of the convolution')
    parser.add_argument('--dilation', default=None, type=str, help='The dilation set of the convolution')
    parser.add_argument('--channel', default=None, type=int, help='The channel number of the network')
    parser.add_argument('--stage', default=None, type=int, help='The stage of the network')
    parser.add_argument('--n_type', default=None, type=int, help='The network architecture of rotation branch 0 - deconv 1- avgpool')
    parser.add_argument('--rotation_type', default=None, type=str, help='The type of rotations, including 6d, 5d, q, eular')
    parser.add_argument('--translation', default=None, type=int, help='If we want to use translation in the network, 0 - No, 1 - Yes')
    parser.add_argument('--confidence', default=None, type=int, help='If we want to use confidence map in the network, 0 - No, 1 - Yes')
    parser.add_argument('--contact', default=None, type=int, help='If we want to use foot contact in the network, 0 - No, 1 - Yes')

    parser.add_argument('--kernel_size_stage_1', default=None, type=str, help='The kernel_size set of the convolution')
    parser.add_argument('--stride_stage_1', default=None, type=str, help='The stride set of the convolution')
    parser.add_argument('--dilation_stage_1', default=None, type=str, help='The dilation set of the convolution')
    parser.add_argument('--kernel_size_stage_2', default=None, type=str, help='The kernel_size set of the convolution')
    parser.add_argument('--n_views', default=4, type=int, help='Number of views to use in Multi-View')
    parser.add_argument('--kernel_width', default=None, type=int, help='Number of views to use in Multi-View')
    parser.add_argument('--padding', default=None, type=int, help='Number of views to use in Multi-View')



    # Training parameters
    parser.add_argument('--train_only_on_cameras', default=None, type=str, help='Specify if limit train to camera idxs')

    parser.add_argument('--lr', default=0.001, type=float, help='The learning rate in the training')
    parser.add_argument('--batch_size', default=128, type=int, help='The batch size')
    parser.add_argument('--train_frames', default=0, type=int, help='The frames number for a training clip, 0 mean random number')
    parser.add_argument('--loss_terms', default='0100', type=str, help='The loss in training we want to use for [foot_contact, 3d_pose, 2d_pose, adversarial] we want to use, 0 - No, 1 - Yes, like: 11111')
    parser.add_argument('--augmentation_terms', default='00', type=str, help='Data augmentation in training we want to use for [pose_flip, projection_depth], 0 - No, 1 - Yes, like: 11')
    parser.add_argument('--save_dir', default=None, type=str, help='Base directory to save network')

    args = parser.parse_args()
    if args.resume:
        print('Loading Config file from checkpoint...')
        config = torch.load(args.resume)['config']
        config.trainer.checkpoint_dir = config.trainer.checkpoint_dir + "/continue"
        config.device = str(args.device)
    elif args.config:
        config = config_parse(args)
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.clearml_logger:
        clearml_logger = get_clearml_logger(config)
    else:
        clearml_logger = None

    print(f'args.device: {str(args.device)}')
    config.device = str(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print(config)
    print(f'Checkpoints will be saved at: {config.trainer.checkpoint_dir}')
    train(config, args.resume)

