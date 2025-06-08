# PaCo: Parametric Point Cloud Completion

**CVPR 2025**

[![Website](https://img.shields.io/badge/%F0%9F%A4%8D%20Project%20-Website-blue)](https://parametric-completion.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2503.08363)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-yellow)](https://huggingface.co/chenzhaiyu/paco)
[![Colab Demo](https://img.shields.io/badge/Colab-Demo-FF6F00?logo=googlecolab&logoColor=yellow)](https://colab.research.google.com/github/parametric-completion/paco/blob/main/demo/demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://raw.githubusercontent.com/parametric-completion/paco/main/LICENSE)

-----

PaCo implements parametric completion, a new point cloud completion paradigm that recovers parametric primitives rather than individual points, for **polygonal surface reconstruction**.

<p align="center">
  <img src="assets/teaser.gif" alt="teaser" width="650px">
</p>

## 🤹‍♂️ Demo

Simply click the badge below to run the demo:

[<img src="https://colab.research.google.com/assets/colab-badge.svg" height="24"/>](https://colab.research.google.com/github/parametric-completion/paco/blob/main/demo/demo.ipynb)

## 🛠️ Setup

### Prerequisites

Before you begin, ensure that your system has the following prerequisites installed:
* Conda
* CUDA Toolkit
* gcc & g++

The code has been tested with Python 3.10, PyTorch 2.6.0 and CUDA 11.8.

### Installation

1. **Clone the repository and enter the project directory:**
   
   ```bash
   git clone https://github.com/parametric-completion/paco && cd paco
   ```

3. **Install dependencies:**
   
   Create a conda environment with all required dependencies:
   ```bash
   . install.sh
   ```

## 🚀 Usage

* Download the preprocessed ABC data: [<img src="https://img.shields.io/badge/OneDrive-blue"/>](https://1drv.ms/u/s!AseUjD457t0Sg-gwSKQ4cC9QIU3jvg) to `./data/abc`:
  
   ```bash
   python ./scripts/download_data.py
   ```

* (Optional) Download pretrained weights: [<img src="https://img.shields.io/badge/OneDrive-blue"/>](https://1drv.ms/f/s!AseUjD457t0Sg-Zwn4_-eHKu8NKIWg?e=fhSKvn) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20-yellow)](https://huggingface.co/chenzhaiyu/paco) to `./ckpt/ckpt-best.pth`:
  
   ```bash
   python ./scripts/download_ckpt.py
   ```

### 🎯 Training

* Start training using one of the two parallelization:

   **Distributed Data Parallel (DDP):**
  
    ```bash
    # Replace device IDs with your own
    CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh
    ```

   **Data Parallel (DP):**
  
    ```bash
    # Replace device IDs with your own
    CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_dp.sh
    ```

* Monitor training progress using TensorBoard:
  
  ```bash
  # Replace ${exp_name} with your experiment name (e.g., default)
  # Board typically available at http://localhost:6006
  tensorboard --logdir './output/${exp_name}/tensorboard'
  ```

### 📊 Evaluation

* Start evaluation of the reconstruction:
  
   ```bash
   # Default checkpoint at `./ckpt/ckpt-best.pth`
   CUDA_VISIBLE_DEVICES=0,1 ./scripts/test.sh
   ```

   The results will be saved to `${output_dir}/evaluation.csv`.

### ⚙️ Available configurations

```bash
# Check available configurations for training
python train.py --cfg job

# Check available configurations for evaluation
python test.py --cfg job
```

Alternatively, review the main configuration file: `conf/config.yaml`.

## 🚧 TODOs

- [x] Demo and pretrained weights
- [x] Dataset and evaluation script
- [x] Hugging Face model

## 🎓 Citation

If you use PaCo in a scientific work, please consider citing the paper:

<a href="https://arxiv.org/pdf/2503.08363"><img class="image" align="left" width="190px" src="./assets/paper_thumbnail.png"></a>
<a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Parametric_Point_Cloud_Completion_for_Polygonal_Surface_Reconstruction_CVPR_2025_paper.pdf">[paper]</a>&nbsp;&nbsp;<a href="https://openaccess.thecvf.com/content/CVPR2025/supplemental/Chen_Parametric_Point_Cloud_CVPR_2025_supplemental.pdf">[supplemental]</a>&nbsp;&nbsp;<a href="https://arxiv.org/abs/2503.08363">[arxiv]</a>&nbsp;&nbsp;<a href="./CITATION.bib">[bibtex]</a><br>
```bibtex
@InProceedings{chen2025paco,
    title = {Parametric Point Cloud Completion for Polygonal Surface Reconstruction}, 
    author = {Zhaiyu Chen and Yuqing Wang and Liangliang Nan and Xiao Xiang Zhu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2025},
    pages = {11749-11758}
}
```
<br clear="left"/>

## 🙏 Acknowledgements

Part of our implementation is based on the [PoinTr](https://github.com/yuxumin/PoinTr) repository. We thank the authors for open-sourcing their great work.
