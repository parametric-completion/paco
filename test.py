import torch
import hydra
from omegaconf import DictConfig

from utils.solver import reconstruct
from utils.evaluator import generate_stats
from utils.logger import get_root_logger
from utils.config import create_reconstruction_result_dir


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def test(cfg: DictConfig):
    """Run the PaCo reconstruction pipeline.
    
    Args:
        cfg: Hydra configuration object containing all test parameters
    """
    # Set up logger
    logger = get_root_logger(name=cfg.log_name)
    logger.info("Running test pipeline...")
    
    # Set up device
    if cfg.use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        assert not cfg.distributed, "Distributed testing is not supported."

    # Configure dataset for testing
    cfg.dataset.bs = 1

    # Prepare output directories for results and logs
    create_reconstruction_result_dir(cfg)

    # Execute the reconstruction process
    logger.info("Starting reconstruction...")
    reconstruct(cfg)

    # Calculate evaluation metrics
    logger.info("Generating evaluation stats...")
    generate_stats(cfg)

    logger.info("Test pipeline completed.")


if __name__ == "__main__":
    test()
