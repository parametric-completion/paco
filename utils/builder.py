import os

import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler

from .data_utils import build_dataset_from_cfg
from .logger import print_log
from .misc import build_lambda_sche, GradualWarmupScheduler, build_lambda_bnsche, worker_init_fn


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg: Model configuration
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    from paco import MODELS

    return MODELS.build(cfg, **kwargs)


def dataset_builder(args, config, logger=None):
    """Build dataset and dataloader from configuration.
    
    Args:
        args: Command line arguments containing distributed training info
        config: Dataset configuration with parameters like batch size
        
    Returns:
        tuple: (sampler, dataloader) for the specified dataset
    """
    dataset = build_dataset_from_cfg(config)
    shuffle = config.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs if shuffle else 1,
                                                 num_workers=int(args.num_workers),
                                                 drop_last=config.subset == 'train',
                                                 worker_init_fn=worker_init_fn,
                                                 sampler=sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs if shuffle else 1,
                                                 shuffle=shuffle,
                                                 drop_last=config.subset == 'train',
                                                 num_workers=int(args.num_workers),
                                                 worker_init_fn=worker_init_fn)
    if logger is not None:
        print_log(f'[DATASET] {config.subset} set with {len(dataset)} samples, batch size: {config.bs}, shuffle: {shuffle}', logger=logger)
    return sampler, dataloader


def model_builder(config):
    """Build model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        nn.Module: Instantiated model
    """
    model = build_model_from_cfg(config)
    return model


def build_optimizer(base_model, config):
    """Build optimizer for the model based on configuration.
    
    Args:
        base_model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            """Add weight decay to specific parameters.
            
            Excludes bias and 1D parameters from weight decay.
            
            Args:
                model: Model to apply weight decay
                weight_decay: Weight decay factor
                skip_list: List of parameter names to skip
                
            Returns:
                list: Parameter groups with appropriate weight decay settings
            """
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    return optimizer


def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    """Build learning rate scheduler based on configuration.
    
    Args:
        base_model: Model being trained
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        last_epoch: Last epoch number for resuming training
        
    Returns:
        object: Learning rate scheduler or list of schedulers
    """
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **sche_config.kwargs_2)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config.kwargs.t_max,
                                      lr_min=sche_config.kwargs.min_lr,
                                      warmup_t=sche_config.kwargs.initial_epochs,
                                      t_in_epochs=True)
    else:
        raise NotImplementedError()

    # Add batch norm momentum scheduler if specified
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)
        scheduler = [scheduler, bnscheduler]

    return scheduler


def resume_model(base_model, args, logger=None):
    """Resume model from checkpoint.
    
    Args:
        base_model: Model to load weights into
        args: Arguments containing experiment path
        logger: Optional logger for printing information
        
    Returns:
        tuple: (start_epoch, best_metrics) from the checkpoint
    """
    ckpt_path = os.path.join(args.output_dir, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0
    print_log(f'[RESUME] Loading model weights from {ckpt_path}...', logger=logger)

    # Load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % int(os.environ["LOCAL_RANK"])}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # Load base model parameters
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # Get training state
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics

    print_log(f'[RESUME] resume checkpoint @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})',
              logger=logger)
    return start_epoch, best_metrics


def resume_optimizer(optimizer, args, logger=None):
    """Resume optimizer state from checkpoint.
    
    Args:
        optimizer: Optimizer to restore state
        args: Arguments containing experiment path
        logger: Optional logger for printing information
        
    Returns:
        int: Status code (0 on success)
    """
    ckpt_path = os.path.join(args.output_dir, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0, 0
    print_log(f'[RESUME] Loading optimizer from {ckpt_path}...', logger=logger)
    # Load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # Restore optimizer state
    optimizer.load_state_dict(state_dict['optimizer'])
    return 0


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=None):
    """Save model checkpoint.
    
    Args:
        base_model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        metrics: Current performance metrics
        best_metrics: Best performance metrics so far
        prefix: Checkpoint filename prefix
        args: Arguments containing output_dir
        logger: Optional logger for printing information
    """
    if int(os.environ["LOCAL_RANK"]) == 0:
        torch.save({
            'base_model': base_model.module.state_dict() if args.distributed else base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics if metrics is not None else dict(),
            'best_metrics': best_metrics if best_metrics is not None else dict(),
        }, os.path.join(args.output_dir, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.output_dir, prefix + '.pth')}", logger=logger)


def load_model(base_model, ckpt_path, logger=None):
    """Load model weights from checkpoint.
    
    Args:
        base_model: Model to load weights into
        ckpt_path: Path to the checkpoint file
        logger: Optional logger for printing information
        
    Raises:
        NotImplementedError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint weights don't match model structure
    """
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('No checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger=logger)

    # Load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # Extract model weights from checkpoint
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError("Mismatch of checkpoint's weights")
    base_model.load_state_dict(base_ckpt)

    # Extract additional information if available
    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    if state_dict.get('metrics') is not None and state_dict.get('best_metrics') is not None:
        print_log(f'Checkpoint @ {epoch} epoch with {str(metrics):s}', logger=logger)
    return
