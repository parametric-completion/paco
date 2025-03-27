import os
import multiprocessing
import io
import sys
import copy

import numpy as np
import torch
import torch.nn.functional as F
import polyfit

from utils import builder


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


def solve_polyfit(input_file, obj_file):
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

    mesh = polyfit.reconstruct(point_cloud, polyfit.SCIP)
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
    solve_polyfit(vg_file, obj_file)

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

    # Process results in parallel
    with multiprocessing.Pool(processes=cfg.num_workers) as pool:
        pool.starmap(solve, tasks)
