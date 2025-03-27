import random
import os
from collections import abc

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pointnet2_ops import pointnet2_utils


def jitter_points(pc, std=0.01, clip=0.05):
    """
    Add random jitter to point cloud data.
    
    Parameters
    ----------
    pc : torch.Tensor
        Point cloud tensor of shape (batch_size, num_points, 3+)
    std : float, optional
        Standard deviation of the noise to be added. Default: 0.01
    clip : float, optional
        Maximum absolute value of the noise. Default: 0.05
        
    Returns
    -------
    torch.Tensor
        Jittered point cloud with same shape as input
    """
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc


def random_sample(data, number):
    """
    Randomly sample a subset of points from each point cloud in the batch.
    
    Parameters
    ----------
    data : torch.Tensor
        Input point cloud tensor of shape (batch_size, num_points, feature_dim)
    number : int
        Number of points to sample from each point cloud
        
    Returns
    -------
    torch.Tensor
        Sampled point cloud of shape (batch_size, number, feature_dim)
        
    Notes
    -----
    The input point cloud must have more points than the number to sample.
    """
    assert data.size(1) > number
    assert len(data.shape) == 3
    ind = torch.multinomial(torch.rand(data.size()[:2]).float(), number).to(data.device)
    data = torch.gather(data, 1, ind.unsqueeze(-1).expand(-1, -1, data.size(-1)))
    return data


def fps(data, number):
    """
    Farthest point sampling algorithm to sample a subset of points.
    
    FPS ensures more uniform coverage compared to random sampling by
    iteratively selecting the point farthest from the already selected points.
    
    Parameters
    ----------
    data : torch.Tensor
        Input point cloud tensor of shape (batch_size, num_points, 3)
    number : int
        Number of points to sample from each point cloud
        
    Returns
    -------
    torch.Tensor
        Sampled point cloud of shape (batch_size, number, 3)
    """
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


def worker_init_fn(worker_id):
    """
    Initialize the random seed for each worker in PyTorch DataLoader.
    
    This function should be passed to the DataLoader's worker_init_fn parameter
    to ensure different workers use different random seeds.
    
    Parameters
    ----------
    worker_id : int
        ID of the DataLoader worker
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_lambda_sche(opti, config, last_epoch=-1):
    """
    Build a lambda learning rate scheduler.
    
    Parameters
    ----------
    opti : torch.optim.Optimizer
        Optimizer to schedule
    config : object
        Configuration object containing scheduler parameters:
        - decay_step: steps after which to decay the learning rate
        - lr_decay: factor by which to decay the learning rate
        - lowest_decay: minimum decay factor
        - warmingup_e: optional, number of warming up epochs
    last_epoch : int, optional
        Index of last epoch. Default: -1
        
    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler
        
    Raises
    ------
    NotImplementedError
        If decay_step is not specified in config
    """
    if config.get('decay_step') is not None:
        # lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        warming_up_t = getattr(config, 'warmingup_e', 0)
        lr_lbmd = lambda e: max(config.lr_decay ** ((e - warming_up_t) / config.decay_step),
                                config.lowest_decay) if e >= warming_up_t else max(e / warming_up_t, 0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return scheduler


def build_lambda_bnsche(model, config, last_epoch=-1):
    """
    Build a batch normalization momentum scheduler.
    
    Parameters
    ----------
    model : nn.Module
        Model containing batch normalization layers
    config : object
        Configuration object containing scheduler parameters:
        - decay_step: steps after which to decay the momentum
        - bn_momentum: initial momentum value
        - bn_decay: factor by which to decay the momentum
        - lowest_decay: minimum decay factor
    last_epoch : int, optional
        Index of last epoch. Default: -1
        
    Returns
    -------
    BNMomentumScheduler
        Batch normalization momentum scheduler
        
    Raises
    ------
    NotImplementedError
        If decay_step is not specified in config
    """
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return bnm_scheduler


def set_random_seed(seed, deterministic=False):
    """
    Set random seed for all random number generators.
    
    This function sets the seed for Python's random, NumPy, and PyTorch
    to ensure reproducibility of results.
    
    Parameters
    ----------
    seed : int
        Seed to be used for all random number generators
    deterministic : bool, optional
        Whether to set CUDNN to deterministic mode. This may impact performance:
        - If True: more reproducible, but potentially slower
        - If False: less reproducible, but potentially faster
        Default: False
    
    Notes
    -----
    When deterministic is True:
    - torch.backends.cudnn.deterministic = True
    - torch.backends.cudnn.benchmark = False
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """
    Check whether a sequence consists of items of the expected type.
    
    Parameters
    ----------
    seq : Sequence
        The sequence to be checked
    expected_type : type
        Expected type of sequence items
    seq_type : type, optional
        Expected sequence type. If None, any Sequence type is acceptable.
        Default: None
        
    Returns
    -------
    bool
        True if seq is a sequence of the expected type, False otherwise
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    """
    Create a function that sets the momentum of batch normalization layers.
    
    Parameters
    ----------
    bn_momentum : float
        Momentum value to set for batch normalization layers
        
    Returns
    -------
    function
        Function that takes a module and sets its batch norm momentum if applicable
    """
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    """
    Scheduler to update batch normalization momentum during training.
    
    This class provides functionality to gradually change the momentum
    of batch normalization layers in a model according to a provided
    momentum function.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model containing batch normalization layers
    bn_lambda : callable
        Function that takes the epoch number and returns the momentum value
    last_epoch : int, optional
        Last epoch number. Default: -1
    setter : callable, optional
        Function that takes momentum value and returns a function to set it
        Default: set_bn_momentum_default
        
    Raises
    ------
    RuntimeError
        If model is not a PyTorch nn.Module
    """

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        """
        Update the momentum for the next epoch.
        
        Parameters
        ----------
        epoch : int, optional
            Epoch to use for momentum calculation.
            If None, use last_epoch + 1. Default: None
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        """
        Get the momentum value for a given epoch.
        
        Parameters
        ----------
        epoch : int, optional
            Epoch to calculate momentum for.
            If None, use last_epoch + 1. Default: None
            
        Returns
        -------
        float
            Momentum value for the specified epoch
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def seprate_point_cloud(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    """
    Separate a point cloud into two parts based on distance to a center point.
    
    This function is used to generate incomplete point clouds by removing a portion
    of points closest to a selected center point.
    
    Parameters
    ----------
    xyz : torch.Tensor
        Input point cloud tensor of shape (batch_size, num_points, 3)
    num_points : int
        Total number of points expected in each point cloud
    crop : int or list
        If int: Number of points to crop from each point cloud
        If list: Range [min, max] for random number of points to crop
    fixed_points : torch.Tensor or list, optional
        Points to use as fixed cropping centers. If None, random centers are used.
        Default: None
    padding_zeros : bool, optional
        If True, cropped points are zeroed out instead of removed from tensor.
        Default: False
        
    Returns
    -------
    tuple of torch.Tensor
        - input_data: Point clouds with points removed/zeroed (B, N-crop, 3) or (B, N, 3)
        - crop_data: Cropped points (B, crop, 3)
    """
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        # Select center point for cropping (random or fixed)
        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        # Calculate distances from center point to all points
        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1)  # 1 1 2048

        # Sort points by distance from center
        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # 2048

        if padding_zeros:
            # Zero out the closest points instead of removing them
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            # Keep only the points beyond the cropping distance
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)  # 1 N 3

        # Get the cropped points
        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        # Apply FPS if crop is a range to ensure same-sized outputs
        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    return input_data.contiguous(), crop_data.contiguous()


def get_ptcloud_img(ptcloud):
    """
    Convert a point cloud to a matplotlib figure image.
    
    Parameters
    ----------
    ptcloud : numpy.ndarray
        Input point cloud of shape (3, N) where the dimensions are x, z, y
        
    Returns
    -------
    numpy.ndarray
        RGB image of the rendered point cloud
    """
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    try:
        ax = fig.gca(projection=Axes3D.name, adjustable='box')
    except:
        ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(30, 45)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    # Convert Matplotlib figure to numpy array
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def visualize_KITTI(path, data_list, titles=['input', 'pred'], cmap=['bwr', 'autumn'], zdir='y',
                    xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    """
    Visualize and save KITTI point cloud data.
    
    Parameters
    ----------
    path : str
        Path where to save the visualization and data files
    data_list : list of torch.Tensor
        List of point clouds to visualize, typically [input, prediction]
    titles : list of str, optional
        Titles for each subplot. Default: ['input', 'pred']
    cmap : list of str, optional
        Colormaps for each subplot. Default: ['bwr', 'autumn']
    zdir : str, optional
        Direction for the z-axis in the plot. Default: 'y'
    xlim, ylim, zlim : tuple, optional
        Axis limits for the visualization. Default: (-1, 1)
        
    Notes
    -----
    This function saves both the visualization as PNG and the raw data as NPY files.
    """
    fig = plt.figure(figsize=(6 * len(data_list), 6))
    cmax = data_list[-1][:, 0].max()

    for i in range(len(data_list)):
        # Special handling for the prediction data (skip the last 2048 points if i=1)
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:, 0] / cmax
        ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color, vmin=-1, vmax=1, cmap=cmap[0], s=4,
                       linewidth=0.05, edgecolors='black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Save visualization as PNG and data as NPY files
    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    """
    Randomly drop points from a point cloud based on the epoch number.
    
    As training progresses (higher epoch numbers), fewer points are kept.
    This implements a form of curriculum learning.
    
    Parameters
    ----------
    pc : torch.Tensor
        Input point cloud tensor of shape (batch_size, num_points, 3)
    e : int
        Current epoch number, used to determine how many points to keep
        
    Returns
    -------
    torch.Tensor
        Point cloud with dropped points, padded with zeros to maintain shape
    """
    # Calculate number of points to keep, decreasing as epochs increase
    up_num = max(64, 768 // (e // 50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1, 1))[0, 0]
    # Sample a subset of points using farthest point sampling
    pc = fps(pc, random_num)
    # Pad with zeros to maintain the original size (2048 points)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim=1)
    return pc


def random_scale(partial, gt, scale_range=[0.8, 1.2]):
    """
    Apply random scaling to both partial point cloud and ground truth.
    
    This function is used for data augmentation by randomly scaling
    the input point clouds within the specified range.
    
    Parameters
    ----------
    partial : torch.Tensor
        Partial point cloud tensor
    gt : torch.Tensor
        Ground truth point cloud tensor
    scale_range : list, optional
        Minimum and maximum scaling factor [min, max]. Default: [0.8, 1.2]
        
    Returns
    -------
    tuple of torch.Tensor
        Scaled partial and ground truth point clouds
    """
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale, gt * scale


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up (increasing) learning rate scheduler.
    
    This scheduler was proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    It gradually increases the learning rate from a small value to the target value over
    a specified number of epochs, then hands over to another scheduler.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer to adjust learning rates for
    multiplier : float
        Target learning rate multiplier:
        - multiplier > 1.0: target_lr = base_lr * multiplier
        - multiplier = 1.0: target_lr = base_lr, starting from 0
    total_epoch : int
        Number of epochs over which to increase learning rate
    after_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler to use after the warmup period. Default: None
        
    Raises
    ------
    ValueError
        If multiplier is less than 1.0
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """
        Calculate learning rates for the current epoch.
        
        Returns
        -------
        list of float
            Learning rates for each parameter group
        """
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        """
        Special handling for ReduceLROnPlateau scheduler.
        
        Parameters
        ----------
        metrics : float
            Performance metric to determine if learning rate should be reduced
        epoch : int, optional
            Current epoch number. Default: None (increments last_epoch)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        """
        Update learning rate for the next epoch.
        
        Handles both standard schedulers and ReduceLROnPlateau.
        
        Parameters
        ----------
        epoch : int, optional
            Current epoch number. Default: None (increments last_epoch)
        metrics : float, optional
            Performance metric for ReduceLROnPlateau. Default: None
        """
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
