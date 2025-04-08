import os
import multiprocessing

import numpy as np
from omegaconf import DictConfig


def start_process_pool(worker_function, parameters, num_processes, timeout=None):
    """
    Start a process pool to execute worker function with given parameters.
    
    Parameters
    ----------
    worker_function: callable
        Function to be executed in parallel
    parameters: list
        List of parameter tuples to pass to the worker function
    num_processes: int
        Number of processes to use
    timeout: int, optional
        Maximum time to wait for processes to complete
        
    Returns
    -------
    list or None
        Results from worker function calls, or None if parameters is empty
    """
    if len(parameters) > 0:
        if num_processes <= 1:
            print('Running loop for {} with {} calls on {} workers'.format(
                str(worker_function), len(parameters), num_processes))
            results = []
            for c in parameters:
                results.append(worker_function(*c))
            return results
        print('Running loop for {} with {} calls on {} subprocess workers'.format(
            str(worker_function), len(parameters), num_processes))
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.starmap(worker_function, parameters)
            return results
    else:
        return None

def _chamfer_distance_single_file(file_in, file_ref, samples_per_model, num_processes=1):
    """
    Calculate chamfer distance between two mesh files.
    
    Parameters
    ----------
    file_in: str
        Path to the input mesh file
    file_ref: str
        Path to the reference mesh file
    samples_per_model: int
        Number of points to sample on each mesh
    num_processes: int, default=1
        Number of processes for KDTree queries
        
    Returns
    -------
    tuple
        (input_file_path, reference_file_path, chamfer_distance)
    """
    import trimesh
    import trimesh.sample
    import sys
    import scipy.spatial as spatial

    def sample_mesh(mesh_file, num_samples):
        """
        Sample points on mesh surface.
        
        Parameters
        ----------
        mesh_file: str
            Path to mesh file
        num_samples: int
            Number of points to sample
            
        Returns
        -------
        numpy.ndarray
            Sampled points, shape (num_samples, 3)
        """
        try:
            mesh = trimesh.load(mesh_file, process=False)
            samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
            return samples
        except:
            return np.zeros((0, 3))
       

    new_mesh_samples = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, 1.0

    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
    kdtree_new_mesh_samples = spatial.cKDTree(new_mesh_samples, leaf_size)
    kdtree_ref_mesh_samples = spatial.cKDTree(ref_mesh_samples, leaf_size)

    ref_new_dist, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, 1, workers=num_processes)
    new_ref_dist, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, 1, workers=num_processes)

    ref_new_dist_sum = np.sum(ref_new_dist)
    new_ref_dist_sum = np.sum(new_ref_dist)
    chamfer_dist = (ref_new_dist_sum + new_ref_dist_sum) / samples_per_model

    return file_in, file_ref, chamfer_dist

def _hausdorff_distance_single_file(file_in, file_ref, samples_per_model):
    """
    Calculate Hausdorff distance between two mesh files.
    
    Parameters
    ----------
    file_in: str
        Path to the input mesh file
    file_ref: str
        Path to the reference mesh file
    samples_per_model: int
        Number of points to sample on each mesh
        
    Returns
    -------
    tuple
        (input_file_path, reference_file_path, distance_new_to_ref, distance_ref_to_new, max_distance)
    """
    import scipy.spatial as spatial
    import trimesh
    import trimesh.sample

    def sample_mesh(mesh_file, num_samples):
        """
        Sample points on mesh surface.
        
        Parameters
        ----------
        mesh_file: str
            Path to mesh file
        num_samples: int
            Number of points to sample
            
        Returns
        -------
        numpy.ndarray
            Sampled points, shape (num_samples, 3)
        """
        try:
            mesh = trimesh.load(mesh_file, process=False)
            samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
            return samples
        except:
            return np.zeros((0, 3))
        

    new_mesh_samples = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, 1.0, 1.0, 1.0

    dist_new_ref, _, _ = spatial.distance.directed_hausdorff(new_mesh_samples, ref_mesh_samples)
    dist_ref_new, _, _ = spatial.distance.directed_hausdorff(ref_mesh_samples, new_mesh_samples)
    dist = max(dist_new_ref, dist_ref_new)
    
    return file_in, file_ref, dist_new_ref, dist_ref_new, dist

def _normal_consistency(file_in, file_ref, samples_per_model, num_processes=1):
    """
    Calculate normal consistency between two mesh files.
    
    Parameters
    ----------
    file_in: str
        Path to the input mesh file
    file_ref: str
        Path to the reference mesh file
    samples_per_model: int
        Number of points to sample on each mesh
    num_processes: int, default=1
        Number of processes for KDTree queries
        
    Returns
    -------
    tuple
        (input_file_path, reference_file_path, normal_consistency_score)
    """
    import scipy.spatial as spatial
    import trimesh
    import sys
    import trimesh.sample
    
    def sample_mesh(mesh_file, num_samples):
        """
        Sample points and normals on mesh surface.
        
        Parameters
        ----------
        mesh_file: str
            Path to mesh file
        num_samples: int
            Number of points to sample
            
        Returns
        -------
        tuple
            (sampled_points, sampled_normals)
        """
        try:
            mesh = trimesh.load(mesh_file, process=False)
            samples, sample_face_indices = trimesh.sample.sample_surface(mesh, num_samples)
            face_normals = np.array(mesh.face_normals)
            normals = face_normals[sample_face_indices]
            return samples, normals
        except:
            return np.zeros((0, 3)), np.zeros((0, 3))
        

    new_mesh_samples, new_normals = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples, ref_normals = sample_mesh(file_ref, samples_per_model)
   

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, 0.0

    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
    kdtree_new_mesh_samples = spatial.cKDTree(new_mesh_samples, leaf_size)
    kdtree_ref_mesh_samples = spatial.cKDTree(ref_mesh_samples, leaf_size)

    _, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, 1, workers=num_processes)
    _, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, 1, workers=num_processes)

    normals_dot_pred_gt = (np.abs(np.sum(new_normals * ref_normals[corr_ref_ids], axis=1)).mean())
    normals_dot_gt_pred = (np.abs(np.sum(ref_normals * new_normals[corr_new_ids], axis=1)).mean() )

    normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2

    return file_in, file_ref, normal_consistency

def mesh_comparison(new_meshes_dir_abs, ref_meshes_dir_abs,
                    num_processes, report_name, samples_per_model=10000, dataset_file_abs=None):
    """
    Compare meshes in two directories and calculate metrics.
    
    This function calculates Hausdorff distance, Chamfer distance, and normal consistency
    between corresponding meshes in the input and reference directories.
    
    Parameters
    ----------
    new_meshes_dir_abs: str
        Path to directory containing input meshes
    ref_meshes_dir_abs: str
        Path to directory containing reference meshes
    num_processes: int
        Number of processes to use for parallel computation
    report_name: str
        Path to output CSV report file
    samples_per_model: int, default=10000
        Number of points to sample on each mesh
    dataset_file_abs: str, optional
        Path to file listing specific models to evaluate
        
    Returns
    -------
    list
        Results of mesh comparisons
    """
    if not os.path.isdir(new_meshes_dir_abs):
        print('Warning: dir to check doesn\'t exist'.format(new_meshes_dir_abs))
        return

    new_mesh_files = [f for f in os.listdir(new_meshes_dir_abs)
                      if os.path.isfile(os.path.join(new_meshes_dir_abs, f))]
    ref_mesh_files = [f for f in os.listdir(ref_meshes_dir_abs)
                      if os.path.isfile(os.path.join(ref_meshes_dir_abs, f))]
    
    if dataset_file_abs is None:
        mesh_files_to_compare_set = set(ref_mesh_files)  # set for efficient search
    else:
        if not os.path.isfile(dataset_file_abs):
            raise ValueError('File does not exist: {}'.format(dataset_file_abs))
        with open(dataset_file_abs) as f:
            mesh_files_to_compare_set = f.readlines()
            mesh_files_to_compare_set = [f.replace('\n', '') + '.obj' for f in mesh_files_to_compare_set]
            mesh_files_to_compare_set = set(mesh_files_to_compare_set)
    

    def ref_mesh_for_new_mesh(new_mesh_file: str, all_ref_meshes: list) -> list:
        """
        Find corresponding reference meshes for a given input mesh.
        
        Parameters
        ----------
        new_mesh_file: str
            Input mesh filename
        all_ref_meshes: list
            List of all reference mesh filenames
            
        Returns
        -------
        list
            Matching reference mesh filenames
        """
        stem_new_mesh_file = new_mesh_file.split('.')[0]
        ref_files = list(set([f for f in all_ref_meshes if f.split('.')[0] == stem_new_mesh_file]))
        return ref_files

    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model))
    if len(call_params) == 0:
        raise ValueError('Results are empty!')
    results_hausdorff = start_process_pool(_hausdorff_distance_single_file, call_params, num_processes)
    results = [(r[0], r[1], str(r[2]), str(r[3]), str(r[4])) for r in results_hausdorff]

    call_params = []
   
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model, 1))
    results_chamfer = start_process_pool(_chamfer_distance_single_file, call_params, num_processes)
    results = [r + (str(results_chamfer[ri][2]),) for ri, r in enumerate(results)]
    
    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model, 1))
    results_normal = start_process_pool(_normal_consistency, call_params, num_processes)
    results = [r + (str(results_normal[ri][2]),) for ri, r in enumerate(results)]
    
    # filter out failed results
    failed_results = [r for r in results if r[-1] == "0.0"]
    results = [r for r in results if r[-1] != "0.0"]

    
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file not in mesh_files_to_compare_set:
            if dataset_file_abs is None:
                new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
                ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
                if len(ref_mesh_files_matching) > 0:
                    reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                    failed_results.append((new_mesh_file_abs, reference_mesh_file_abs, str(2), str(2), str(2), str(2), str(0)))
        else:
            mesh_files_to_compare_set.remove(new_mesh_file)
            
    # no reconstruction but reference
    for ref_without_new_mesh in mesh_files_to_compare_set:
        new_mesh_file_abs = os.path.join(new_meshes_dir_abs, ref_without_new_mesh)
        reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_without_new_mesh)
        failed_results.append((new_mesh_file_abs, reference_mesh_file_abs, str(1), str(1), str(1), str(1), str(0)))

    # sort by file name
    failed_results = sorted(failed_results, key=lambda x: x[0])
    results = sorted(results, key=lambda x: x[0])
    with open(report_name, 'w') as f:
        f.write('in mesh,ref mesh,Hausdorff dist new-ref,Hausdorff dist ref-new,Hausdorff dist,Chamfer dist(1: no input),Normal consistency(0: no input)\n')
        for r in failed_results:
            f.write(','.join(r) + '\n')
        for r in results:
            f.write(','.join(r) + '\n')


    return results
    
def generate_stats(cfg: DictConfig):
    """
    Evaluate hausdorff distance, chamfer distance and normal consistency between reconstructed and GT models.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration containing:
        - reconstruction_dir: Directory with reconstructed meshes
        - reference_dir: Directory with ground truth meshes
        - num_workers: Number of parallel workers
        - csv_path: Output file path
        - evaluate.num_samples: Number of points to sample per mesh
    """
    
    mesh_comparison(
        new_meshes_dir_abs=cfg.reconstruction_dir,
        ref_meshes_dir_abs=cfg.reference_dir,
        num_processes=cfg.num_workers,
        report_name=cfg.csv_path,
        samples_per_model=cfg.evaluate.num_samples,
        dataset_file_abs=None)


if __name__ == '__main__':
    generate_stats()
