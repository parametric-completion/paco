import os

import h5py
import numpy as np


class IO:
    """
    A utility class for reading various file formats including .npy, .pcd, .ply, .h5, and .txt.
    """

    @classmethod
    def get(cls, file_path):
        """
        Read a file based on its extension and return the loaded data.

        Args:
            file_path (str): The path to the file.

        Returns:
            The data loaded from the file.

        Raises:
            Exception: If the file extension is unsupported.
        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        """
        Read a .npy file using numpy.

        Args:
            file_path (str): The path to the .npy file.

        Returns:
            numpy.ndarray: The array loaded from the file.
        """
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        """
        Read a text file containing numerical data using numpy.

        Args:
            file_path (str): The path to the .txt file.

        Returns:
            numpy.ndarray: The array loaded from the file.
        """
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        """
        Read an HDF5 file and return the dataset under the 'data' key.

        Args:
            file_path (str): The path to the .h5 file.

        Returns:
            The dataset loaded from the HDF5 file.
        """
        f = h5py.File(file_path, 'r')
        return f['data'][()]
