# FSHNet
Here is an official implementation for paper "A Hybrid Frequency-Spatial Domain Model for Sparse Image Reconstruction in Scanning Transmission Electron Microscopy" accepted in ICCV 2021.<br>

## Introduction
Scanning transmission electron microscopy (STEM) is a powerful technique in high-resolution atomic imaging of materials. Decreasing scanning time and reducing electron beam exposure with an acceptable signal-to-noise ratio are two popular research aspects when applying STEM to beam-sensitive materials. Specifically, partially sampling with fixed electron doses is one of the most important solutions, and then the lost information is {restored by computational methods}. Following successful applications of deep learning in image in-painting, we have developed an encoder-decoder network to reconstruct STEM images in extremely sparse sampling cases. In our model, we combine both local pixel information from convolution operators and global texture features, by applying specific filter operations on the frequency domain to acquire initial reconstruction and global structure prior. Our method can effectively restore texture structures and be robust in different sampling ratios with Poisson noise. A comprehensive study demonstrates that our method gains about 50\% performance enhancement in comparison with the state-of-art methods. Code is available at https://github.com/icthrm/Sparse-Sampling-Reconstruction.

## Operation System
Ubuntu 18.04 or CentOS7

## Requirements
Python 3.6 <br>
Pytorch 1.7 <br>
opencv-python 4.4 <br>
numpy 1.19 <br>

## Pretrained Models and Dataset
model: pretrained models <br>
data/test/: test data drawn in the manuscript and supplementary materials

## Usage
sh run.sh
#### Generate frequency output
python Preprocess.py

#### For detailed parameter settings, use
python STEM-Prior-Init.py --help

#### Linear initial reconstruction version
Just replace STEM-Prior-Init.py by STEM-Prior-Init-Linear.py in script