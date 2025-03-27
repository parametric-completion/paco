import os
import sys
import random

import numpy as np
import torch.utils.data as data

# Set up the base directory and add it to the system path for local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import data_transforms
from .io import IO
from .build import DATASETS


@DATASETS.register_module()
class ABC(data.Dataset):
    """
    Dataset class for the ABC dataset.

    This dataset loads complete and partial point clouds along with plane data.
    It supports training and testing subsets and applies a sequence of transformations
    to the data.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the ABC dataset.

        Args:
            config: Configuration object containing dataset paths, number of points,
                    number of planes, subset type, and other parameters.
        """
        self.data_root = config.data_path
        self.complete_points_path = config.complete_points_path
        self.complete_planes_path = config.complete_planes_path
        self.input_points_path = config.input_points_path
        self.subset = config.subset

        self.num_points = config.num_points
        self.num_planes = config.num_planes
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            model_id = line.split(".")[0]
            self.file_list.append({
                'model_id': model_id,
                'file_path': line
            })

        # Use 24 renderings for training and 1 for testing
        self.num_renderings = config.num_renderings if self.subset == 'train' else 1

        # Initialize the data transformations
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        """
        Define the sequence of data transformations to be applied on each sample.

        Returns:
            A composed transformation function.
        """
        return data_transforms.Compose([
            {
                'callback': 'UpSamplePlanes',
                'parameters': {
                    'n_planes': self.num_planes
                },
                'objects': ['planes_gt']
            },
            {
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['points_pc']
            },
            {
                'callback': 'ToTensor',
                'objects': ['points_gt', 'planes_gt', 'points_pc']
            }
        ])

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing:
              - model_id: Identifier of the model.
              - A tuple with:
                  - points_gt (first 3 coordinates)
                  - points_gt (last coordinate)
                  - planes_gt (first 3 coordinates)
                  - planes_gt (last coordinate)
                  - points_pc: Partial points data.
        """
        sample = self.file_list[idx]
        data = {}

        # Select a random rendering index for training; use 0 for testing
        render_idx = random.randint(0, self.num_renderings - 1) if self.subset == 'train' else 0

        # Load complete point cloud data
        gt_point_path = os.path.join(self.complete_points_path, sample['model_id'] + '.npy')
        data['points_gt'] = IO.get(gt_point_path).astype(np.float32)

        # Load complete plane data
        gt_plane_path = os.path.join(self.complete_planes_path, sample['model_id'] + '.npy')
        data['planes_gt'] = IO.get(gt_plane_path).astype(np.float32)

        # Load partial point cloud data for the selected rendering index
        input_point_path = os.path.join(self.input_points_path, sample['model_id'] + '_%02d.npy' % render_idx)
        data['points_pc'] = IO.get(input_point_path).astype(np.float32)

        # Ensure the complete points data has the expected number of points
        assert data['points_gt'].shape[0] == self.num_points

        # Apply data transformations if they are defined
        if self.transforms is not None:
            data = self.transforms(data)

        return sample['model_id'], (
            data['points_gt'][..., :3],
            data['points_gt'][..., -1],
            data['planes_gt'][..., :3],
            data['planes_gt'][..., -1],
            data['points_pc']
        )

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.file_list)
