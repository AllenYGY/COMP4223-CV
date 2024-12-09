#!/bin/bash

conda create --name CV python=3.10 -y

conda activate CV
conda install numpy pandas matplotlib opencv -y
conda deactivate

echo "Conda environment 'CV' has been successfully set up."