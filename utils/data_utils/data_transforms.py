# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 14:38:36

import numpy as np
import torch


class Compose(object):
    """
    Composes several transformations together.

    Each transformation is defined by a dictionary that specifies a 'callback' (the name of the
    transformation class as a string), optional 'parameters' for the transformation, and 'objects'
    which lists the data keys to which the transformation should be applied.
    """
    def __init__(self, transforms):
        """
        Initialize the Compose object with a list of transformation dictionaries.

        Args:
            transforms: List of dictionaries. Each dictionary contains:
                - 'callback': Name of the transformation class (as a string)
                - 'parameters': (Optional) Parameters for the transformation
                - 'objects': List of keys in the data to apply this transformation to
        """
        self.transformers = []
        for tr in transforms:
            # Dynamically retrieve the transformation class using eval
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        """
        Apply the composed transformations to the input data.

        Args:
            data: Dictionary containing the data to be transformed

        Returns:
            The transformed data dictionary
        """
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            # Generate a random value (currently not used)
            rnd_value = np.random.uniform(0, 1)

            for k, v in data.items():
                if k in objects and k in data:
                    data[k] = transform(v)
        return data


class ToTensor(object):
    """
    Convert a numpy array to a PyTorch tensor.
    """
    def __init__(self, parameters):
        """
        Initialize the ToTensor transformation.

        Args:
            parameters: Not used for this transformation
        """
        pass

    def __call__(self, arr):
        """
        Convert a numpy array to a PyTorch tensor of type float.

        Args:
            arr: Numpy array to convert

        Returns:
            A PyTorch tensor
        """
        return torch.from_numpy(arr.copy()).float()


class UpSamplePoints(object):
    """
    Upsample a point cloud to a fixed number of points.
    """
    def __init__(self, parameters):
        """
        Initialize the UpSamplePoints transformation.

        Args:
            parameters: Dictionary with key 'n_points' specifying the target number of points
        """
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        """
        Upsample the point cloud.

        If the point cloud has more points than required, a random subset is selected.
        Otherwise, the point cloud is repeatedly tiled until enough points are available,
        and then additional points are selected randomly.

        Args:
            ptcloud: Numpy array representing the point cloud

        Returns:
            A point cloud with exactly self.n_points points
        """
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            # If there are more points than needed, randomly select self.n_points points
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


class UpSamplePlanes(object):
    """
    Upsample plane data to a fixed number of planes.
    """
    def __init__(self, parameters):
        """
        Initialize the UpSamplePlanes transformation.

        Args:
            parameters: Dictionary with key 'n_planes' specifying the target number of planes
        """
        self.n_planes = parameters['n_planes']

    def __call__(self, plane):
        """
        Upsample the plane data.

        If there are fewer planes than required, zero padding is added (with a padding index of -1)
        to reach the target number of planes. If there are more planes, the plane data is truncated.

        Args:
            plane: Numpy array representing the plane data

        Returns:
            A numpy array with exactly self.n_planes rows
        """
        if plane.shape[0] < self.n_planes:
            # Create zero padding for the missing planes
            padding = np.zeros((self.n_planes - plane.shape[0], plane.shape[1] - 1))
            padding_index = np.ones((self.n_planes - plane.shape[0], 1)) * (-1)
            padding = np.concatenate([padding, padding_index], axis=1)
            plane = np.concatenate([plane, padding])
        else:
            plane = plane[:self.n_planes, :]

        return plane
