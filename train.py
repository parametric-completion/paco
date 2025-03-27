import os

import torch
from tensorboardX import SummaryWriter
import hydra
from omegaconf import DictConfig

from utils import dist_utils, misc
from utils.runner import run_trainer
from utils.logger import get_root_logger
from utils.config import create_experiment_dir


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    """Main function to initialize and run the training process."""

    # Check if CUDA is available and enable cuDNN benchmark for performance
    if cfg.use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

        # Initialize distributed training if applicable
        if cfg.distributed:
            dist_utils.init_dist(cfg.launcher)

            # Re-set GPU IDs when using distributed training
            _, world_size = dist_utils.get_dist_info()
            cfg.world_size = world_size
    
    # Retrieve local rank
    local_rank = int(os.environ["LOCAL_RANK"])
    
    create_experiment_dir(cfg)

    # Set up logger
    logger = get_root_logger(name=cfg.log_name)

    # Initialize TensorBoard writers for training and validation
    # Only the main process (local_rank == 0) writes to TensorBoard
    if local_rank == 0:
        train_writer = SummaryWriter(os.path.join(cfg.output_dir, 'tensorboard/train'))
        val_writer = SummaryWriter(os.path.join(cfg.output_dir, 'tensorboard/test'))
    else:
        train_writer = None
        val_writer = None

    # Adjust batch size based on the distributed training setting
    if cfg.distributed:
        assert cfg.total_bs % world_size == 0, "Total batch size must be divisible by world size."
        cfg.dataset.bs = cfg.total_bs // world_size
    else:
        cfg.dataset.bs = cfg.total_bs

    # Log distributed training status
    if local_rank == 0:
        logger.info(f'Distributed training: {cfg.distributed}')

    # Set random seed for reproducibility if provided
    if cfg.seed is not None:
        if local_rank == 0:
            logger.info(f'Set random seed to {cfg.seed}, deterministic: {cfg.deterministic}')
        misc.set_random_seed(cfg.seed + local_rank, deterministic=cfg.deterministic)

    # In distributed mode, confirm local rank matches the distributed rank
    if cfg.distributed:
        assert local_rank == torch.distributed.get_rank(), "Local rank does not match distributed rank."

    # Run trainer
    run_trainer(cfg, train_writer, val_writer)


if __name__ == '__main__':

    train()
