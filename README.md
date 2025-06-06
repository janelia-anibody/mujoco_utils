# `mujoco_utils` 

Common utilities for working with MuJoCo MJCF models

![lint](https://github.com/janelia-anibody/mujoco_utils/actions/workflows/lint.yml/badge.svg)

## Installation
To install into an existing conda environment, follow these steps:
```bash
git clone https://github.com/janelia-anibody/mujoco_utils.git
cd mujoco_utils
pip install -e .
```
Or create a new conda environment (e.g. `anibody`) and then install:
```bash
conda create --name anibody python=3.10 pip
conda activate anibody
git clone https://github.com/janelia-anibody/mujoco_utils.git
cd mujoco_utils
pip install -e .
```
Verify that it works:
```bash
python -c "import mujoco_utils"
```
