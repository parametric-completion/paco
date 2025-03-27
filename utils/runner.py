import copy
import time
import os

import torch
import torch.nn as nn

from utils import builder, dist_utils
from utils.logger import get_logger, print_log
from utils.average_meter import AverageMeter


def init_device(gpu_ids):
    """
    Init devices.

    Parameters
    ----------
    gpu_ids: list of int
        GPU indices to use
    """
    # set multiprocessing sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')

    # does not work for DP after import torch with PyTorch 2.0, but works for DDP nevertheless
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)[1:-1]


def run_trainer(cfg, train_writer=None, val_writer=None):
    """
    Main training function that handles the complete training and validation cycle.

    Args:
        cfg: Configuration object containing all training parameters
        train_writer: TensorBoard writer for training metrics
        val_writer: TensorBoard writer for validation metrics

    Returns:
        None
    """
    logger = get_logger(cfg.log_name)
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Build datasets for training and testing
    train_config = copy.deepcopy(cfg.dataset)
    train_config.subset = "train"
    test_config = copy.deepcopy(cfg.dataset)
    test_config.subset = "test"
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(cfg, train_config, logger=logger), \
        builder.dataset_builder(cfg, test_config, logger=logger)
    
    # Build and initialize the model
    base_model = builder.model_builder(cfg.model)
    if cfg.use_gpu:
        base_model.to(local_rank)

    # Initialize training parameters
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Load checkpoints if resuming training or starting from pretrained model
    if cfg.resume_last:
        start_epoch, best_metrics = builder.resume_model(base_model, cfg, logger=logger)
    elif cfg.resume_from is not None:
        builder.load_model(base_model, cfg.resume_from, logger=logger)

    # Print model information for debugging
    if cfg.debug:
        print_log('Trainable_parameters:', logger=logger)
        print_log('=' * 25, logger=logger)
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger=logger)

        print_log('Untrainable_parameters:', logger=logger)
        print_log('=' * 25, logger=logger)
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger=logger)

    # Set up distributed training if needed
    if cfg.distributed:
        if cfg.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # Set up optimizer and learning rate scheduler
    optimizer = builder.build_optimizer(base_model, cfg)

    if cfg.resume_last:
        builder.resume_optimizer(optimizer, cfg, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, cfg, last_epoch=start_epoch - 1)

    # Main training loop
    base_model.zero_grad()
    for epoch in range(start_epoch, cfg.max_epoch + 1):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        # Initialize timing and loss tracking
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_losses = AverageMeter(
            ["plane_chamfer_loss", "classification_loss", 'chamfer_norm1_loss', 'chamfer_norm2_loss',
             "plane_normal_loss", "repulsion_loss", "total_loss"])

        num_iter = 0

        base_model.train()
        n_batches = len(train_dataloader)
        for idx, (model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            dataset_name = cfg.dataset.name
            if dataset_name == 'ABC':
                gt = data[0].cuda()  # bs, n, 3
                gt_index = data[1].cuda()  # bs, n
                plane = data[2].cuda()  # bs, 20, 3
                plane_index = data[3].cuda()  # bs, 20
                pc = data[4].cuda()  # bs, n, 7 x, y, z, nx, ny, nz, index
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            ret, class_prob = base_model(pc)

            # Calculate losses and backpropagate
            losses = base_model.module.get_loss(cfg.loss, ret, class_prob, gt, gt_index, plane, plane_index)
            _loss = losses['total_loss']
            _loss.backward()

            # Update weights after accumulating gradients
            if num_iter == cfg.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(cfg, 'grad_norm_clip', 10),
                                               norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            # Process and log loss metrics
            if cfg.distributed:
                for key in losses.keys():
                    losses[key] = dist_utils.reduce_tensor(losses[key], cfg)
                train_losses.update([losses[key].item() * 1000 for key in losses.keys()])
            else:
                train_losses.update([losses[key].item() * 1000 for key in losses.keys()])

            if cfg.distributed:
                torch.cuda.synchronize()

            # Log metrics to TensorBoard
            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                for key in losses.keys():
                    train_writer.add_scalar(f'Loss/Batch/{key}', losses[key].item() * 1000, n_itr)

            # Update timing information
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # Print progress information
            if idx % 100 == 0:
                print_log(f'[Epoch {epoch}/{cfg.max_epoch}][Batch {idx + 1}/{n_batches}] | BatchTime = {batch_time.val():.3f}s | '
                          f'Losses = [{", ".join(f"{l:.3f}" for l in train_losses.val())}] | lr = {optimizer.param_groups[0]["lr"]:.6f}', logger=logger)

            # Handle special case for GradualWarmup scheduler
            if cfg.scheduler.type == 'GradualWarmup':
                if n_itr < cfg.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        # Step the learning rate scheduler after each epoch
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        # Log epoch-level training metrics
        if train_writer is not None:
            for i, key in enumerate(losses.keys()):
                train_writer.add_scalar(f'Loss/Epoch/{key}', train_losses.avg(i), epoch)
            print_log(f'[Training] Epoch: {epoch} | EpochTime = {epoch_end_time - epoch_start_time:.3f}s | '
                      f'Losses = [{", ".join(f"{l:.4f}" for l in train_losses.avg())}]', logger=logger)

        # Run validation at specified frequency
        if epoch % cfg.val_freq == 0:
            test_losses = validate(base_model, test_dataloader, epoch, val_writer, cfg, logger=logger)
            if best_metrics is None:
                best_metrics = test_losses[cfg.consider_metric]

            # Save checkpoint if current model is the best so far
            if test_losses[cfg.consider_metric] < best_metrics:
                best_metrics = test_losses[cfg.consider_metric]
                metrics = test_losses
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', cfg,
                                        logger=logger)

        # Save checkpoints
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', cfg, logger=logger)
        if (cfg.max_epoch - epoch) < 2:
            metrics = test_losses
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    cfg, logger=logger)
    
    # Close TensorBoard writers
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, config, logger=None):
    """
    Validate the model on the test dataset.
    
    Args:
        base_model: Model to validate
        test_dataloader: DataLoader for test data
        epoch: Current epoch number
        val_writer: TensorBoard writer for validation metrics
        config: Configuration object containing validation parameters
        logger: Logger for printing information
        
    Returns:
        metrics: Dictionary containing validation metrics
    """
    base_model.eval()

    # Initialize metrics tracking
    test_losses = AverageMeter(
        ["plane_chamfer_loss", "classification_loss", 'chamfer_norm1_loss', 'chamfer_norm2_loss', "plane_normal_loss",
         "repulsion_loss", "total_loss"])
    n_samples = len(test_dataloader)  # bs is 1
    interval = n_samples // 10 + 1

    # Validation loop
    with torch.no_grad():
        for idx, (model_ids, data) in enumerate(test_dataloader):
            model_id = model_ids[0]
            dataset_name = config.dataset.name
            if dataset_name == 'ABC':
                gt = data[0].cuda()
                gt_index = data[1].cuda()
                plane = data[2].cuda()
                plane_index = data[3].cuda()
                pc = data[4].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # Forward pass and loss computation
            ret, class_prob = base_model(pc)
            losses = base_model.module.get_loss(config.loss, ret, class_prob, gt, gt_index, plane, plane_index)
            
            if config.distributed:
                for key in losses.keys():
                    losses[key] = dist_utils.reduce_tensor(losses[key], config)
            test_losses.update([losses[key].item() * 1000 for key in losses.keys()])

        # Synchronize processes in distributed mode
        if config.distributed:
            torch.cuda.synchronize()

    # Log validation metrics to TensorBoard
    if val_writer is not None:
        for i, key in enumerate(losses.keys()):
            val_writer.add_scalar(f'Loss/Epoch/{key}', test_losses.avg(i), epoch)

    print_log(f'[Validation] Epoch: {epoch} | Losses = [{", ".join(f"{l:.4f}" for l in test_losses.avg())}]', logger=logger)
    
    # Prepare metrics dictionary for return
    metrics = {}
    for i, key in enumerate(losses.keys()):
        metrics[key] = test_losses.avg(i)

    return metrics
