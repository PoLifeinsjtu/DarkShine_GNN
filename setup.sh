#!/bin/bash

# initiate the environment
source /sw/anaconda/3.7-2020.02/thisconda.sh 
# conda activate /lustre/collider/zhangyulei/DeepLearning/env
conda activate /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/env

# adding project path to PYTHONPATH
GNNPATH=/lustre/collider/wanghuayang/DeepLearning/tracking/trkgnn
export PYTHONPATH="${PYTHONPATH}:${GNNPATH}"
