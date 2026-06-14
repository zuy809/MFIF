# MFIF

This repository provides MATLAB code and sample datasets for our multi-focus image fusion (MFIF) experiments.

## Contents

- `align_demo.m`: registration demo for misaligned focus-stack images.
- `mfif_fuse.m`: main fusion function.
- `dataset/`: focus-stack images used for testing and reproducibility.
- `ecc.m`, `param_update.m`, `spatial_interp.m`, `warp_jacobian.m`, `image_jacobian.m`, `next_level.m`: supporting functions.

## Dataset

The `dataset` folder includes public MFIF image pairs and representative multi-frame focus stacks, including `ball`, `bucket`, `grayscale`, `kitchen`, `lytro`, `screw nut`, and `standrad ball`.

## Usage

Run the code in MATLAB from the repository root. For example, start with `align_demo.m` for registration, and use `mfif_fuse.m` for fusion after preparing the input image folder.

The code and datasets are provided to support reproducibility of the revised manuscript.
