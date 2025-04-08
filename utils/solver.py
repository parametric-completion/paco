import os
import io
import sys
import copy
from multiprocessing import Pool
from multiprocessing.context import TimeoutError as MPTimeout
from io import StringIO
from wurlitzer import pipes, STDOUT

import numpy as np
import torch
import torch.nn.functional as F

try:
    # in case locally built PolyFit as in `install.sh`
    sys.path.append("./utils/polyfit/release/lib/python")
    import polyfit
except ImportError:
    print("PolyFit not found. Exiting.")
    exit(1)

from utils import builder

def run_tasks_with_timeout(tasks, timeout, num_workers):
    with Pool(processes=num_workers) as pool:
        async_results = []
        for i, task in enumerate(tasks):
            async_results.append(pool.apply_async(solve, task))

        for i, async_result in enumerate(async_results):
            try:
                async_result.get(timeout=timeout)
            except MPTimeout:
                print(f"[Timeout] Task {i} (model {tasks[i][-2]}) timed out.", file=sys.stderr)
                
def save_vg(points, planes, normal, filepath, group_num=1280):
    """
    Save the processed point cloud data into a .vg file.
    
    Parameters
    ----------
    points: np.ndarray
        Array of point coordinates.
    planes: np.ndarray
        Plane parameters.
    normal: np.ndarray
        Normal vectors.
    filepath: str
        Path to save the .vg file.
    group_num: int
        Number of points per group.
    """
    from random import random

    out = ''
    out += f'num_points: {points.shape[0]}\n'
    output = io.StringIO()
    np.savetxt(output, points[:, :3], fmt="%.6f %.6f %.6f")
    out += output.getvalue()
    output.close()

    out += f'num_colors: {points.shape[0]}\n'
    colors = np.ones((points.shape[0], 3)) * 128
    output = io.StringIO()
    np.savetxt(output, colors, fmt="%d %d %d")
    out += output.getvalue()
    output.close()

    out += f'num_normals: {points.shape[0]}\n'
    output = io.StringIO()
    np.savetxt(output, normal, fmt="%.6f %.6f %.6f")
    out += output.getvalue()
    output.close()

    num_groups = planes.shape[0]
    out += f'num_groups: {num_groups}\n'

    j_base = 0
    for i in range(num_groups):
        out += 'group_type: 0\n'
        out += 'num_group_parameters: 4\n'
        out += f'group_parameters: {planes[i][0]} {planes[i][1]} {planes[i][2]} {planes[i][3]}\n'
        out += f'group_label: group_{i}\n'
        out += f'group_color: {random()} {random()} {random()}\n'
        out += f'group_num_point: {group_num}\n'
        out += ' '.join(str(j) for j in range(j_base, j_base + group_num)) + '\n'
        j_base += group_num
        out += 'num_children: 0\n'

    with open(filepath, 'w') as fout:
        fout.writelines(out)


def solve_polyfit(input_file, obj_file, solver="SCIP"): 
    """
    Run PolyFit reconstruction solver and save the result as OBJ file.
    
    Parameters
    ----------
    input_file: str
        Path to input .vg file.
    obj_file: str
        Path to save the output OBJ file.
    """
    polyfit.initialize()
    point_cloud = polyfit.read_point_set(input_file)
    if not point_cloud:
        print(f"Failed loading point cloud from file: {input_file}", file=sys.stderr)
        sys.exit(1)
    out = StringIO()
    if solver == "SCIP" and hasattr(polyfit, "SCIP"):
        with pipes(stdout=out, stderr=STDOUT):
            mesh = polyfit.reconstruct(point_cloud, polyfit.SCIP)
    elif solver == "GUROBI" and hasattr(polyfit, "GUROBI"):
        with pipes(logger, stderr=STDOUT):
            mesh = polyfit.reconstruct(point_cloud, polyfit.GUROBI)
    else:
        raise ValueError(f"Please specify a valid solver: SCIP or GUROBI and make sure it is available.")
    
    if mesh is None:
        raise RuntimeError("Reconstruction failed.")
    else:
        polyfit.save_mesh(obj_file, mesh)


def solve(dense_points, plane_params, class_prob, model_id, cfg):
    """
    Post-process the network output and run reconstruction pipeline.
     
    Parameters
    ----------
    dense_points: np.ndarray
        Dense point cloud from network output.
    plane_params: np.ndarray
        Plane parameters from network output.
    class_prob: np.ndarray
        Class probability scores.
    model_id: str
        Model identifier.
    cfg: DictConfig
        Configuration object containing reconstruction parameters.
    """
    mask = class_prob[:, 0] > 0.7
    dense_points = dense_points.reshape(class_prob.shape[0], -1, 3)[mask].reshape(-1, 3)
    plane_params = plane_params[mask].reshape(-1, 4)
    group_num = cfg.model.num_points // cfg.model.num_queries
    normals = np.repeat(plane_params[:, :3], group_num, axis=0)

    # Save intermediate .vg file for reconstruction
    vg_file = os.path.join(cfg.reconstruction_dir, model_id + '.vg')
    save_vg(dense_points, plane_params, normals, vg_file)

    # Run PolyFit reconstruction to generate mesh
    obj_file = os.path.join(cfg.reconstruction_dir, model_id + '.obj')
    try:
        solve_polyfit(vg_file, obj_file, solver=cfg.evaluate.solver)
    except Exception as e:
        print(f"Failed to reconstruct {model_id}")
        
    # Clean up intermediate files if not needed
    if not cfg.evaluate.keep_vg:
        os.remove(vg_file)


def reconstruct(cfg):
    """
    Run inference and reconstruction on the test dataset.
    
    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration containing model, dataset and reconstruction parameters.
    """
    # Load and prepare model for inference
    base_model = builder.model_builder(cfg.model)
    builder.load_model(base_model, cfg.checkpoint_path)

    base_model.cuda()
    base_model.eval()
    
    if cfg.evaluate.single_file_path is not None:
        file_path = cfg.evaluate.single_file_path
        pc =  np.load(file_path).astype(np.float32)
        pc = torch.from_numpy(pc).float().unsqueeze(0).cuda()
        with torch.no_grad():
            ret, class_prob = base_model(pc)
            class_prob = F.softmax(class_prob, dim=-1)
        solve(ret[-1].squeeze(0).cpu().numpy(), ret[-2].squeeze(0).cpu().numpy(), class_prob.squeeze(0).cpu().numpy(), file_path.split('/')[-1].split('.')[0], cfg)
        return

    # Prepare test dataloader
    test_config = copy.deepcopy(cfg.dataset)
    test_config.subset = "test"
    _, test_dataloader = builder.dataset_builder(cfg, test_config)

    # Perform inference
    tasks = []
    with torch.no_grad():
        for idx, (model_ids, data) in enumerate(test_dataloader):
            model_id = model_ids[0]
            pc = data[4].cuda()
            ret, class_prob = base_model(pc)
            class_prob = F.softmax(class_prob, dim=-1)
            sample = (
                ret[-1].squeeze(0).cpu().numpy(),
                ret[-2].squeeze(0).cpu().numpy(),
                class_prob.squeeze(0).cpu().numpy(),
                model_id,
                cfg
            )
            tasks.append(sample)

    run_tasks_with_timeout(tasks, cfg.evaluate.time_out, cfg.num_workers)
