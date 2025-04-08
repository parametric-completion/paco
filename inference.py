import torch
import hydra
from omegaconf import DictConfig

from utils.solver import reconstruct
from utils.logger import get_root_logger
from utils.config import create_reconstruction_result_dir

    
@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def inference(cfg: DictConfig):
    """Run the PaCo reconstruction pipeline.
    
    Args:
        cfg: Hydra configuration object containing all test parameters
    """
    # Set up logger
    logger = get_root_logger(name=cfg.log_name)
    
    # Set up device
    if cfg.use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        assert not cfg.distributed, "Distributed testing is not supported."

    # Prepare output directories for results and logs
    create_reconstruction_result_dir(cfg)

    assert cfg.evaluate.single_file_path is not None 
    # Execute the reconstruction process
    logger.info("Starting reconstruction...")
    reconstruct(cfg)


if __name__ == "__main__":
    inference()
