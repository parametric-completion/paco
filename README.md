# PaCo: Parametric Point Cloud Completion

-----------
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://parametric-completion.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2503.08363) 

PaCo implements parametric completion, a new point cloud completion paradigm that recovers parametric primitives rather than individual points, for **polygonal surface reconstruction**.

<p align="center">
  <img src="assets/teaser.gif" alt="teaser" width="650px">
</p>


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

2. **Install dependencies:**
   Create a conda environment with all required dependencies:
   ```bash
   . install.sh
   ```

## 🚀 Usage

### 🎯 Training

Start training using one of the two parallelization:

**Distributed Data Parallel (DDP):**
```bash
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh
```

**Data Parallel (DP):**
```bash
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_dp.sh
```

### ⚙️ Available configurations
```bash
# check available configurations for training
python train.py --cfg job

# check available configurations for evaluation
python test.py --cfg job
```

Alternatively, review the main configuration file: `conf/config.yaml`.

## 🚧 TODOs

- [ ] Pretrained weights
- [ ] Dataset and evaluation script
- [ ] Hugging Face space

## 🎓 Citation

If you use PaCo in a scientific work, please consider citing the paper:

<a href="https://arxiv.org/pdf/2503.08363"><img class="image" align="left" width="150px" src="./assets/paper_thumbnail.png"></a>
<a href="https://arxiv.org/pdf/2503.08363">[paper]</a>&nbsp;&nbsp;<a href="https://arxiv.org/abs/2503.08363">[arxiv]</a><br>
```bibtex
@InProceedings{chen2025paco,
    title={Parametric Point Cloud Completion for Polygonal Surface Reconstruction}, 
    author={Zhaiyu Chen and Yuqing Wang and Liangliang Nan and Xiao Xiang Zhu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```
<br clear="left"/>

## 🙏 Acknowledgements

Part of our implementation is based on the [PoinTr](https://github.com/yuxumin/PoinTr) repository. We appreciate the authors for open-sourcing their great work.
