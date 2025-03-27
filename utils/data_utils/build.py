from utils import registry


# Create a registry for datasets
DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args=None):
    """
    Build a dataset from configuration

    Args:
        cfg (eDICT): Configuration dictionary specifying the dataset parameters
        default_args: Optional default arguments for the dataset

    Returns:
        Dataset: A constructed dataset specified by the configuration
    """
    return DATASETS.build(cfg, default_args=default_args)
