#!/bin/bash
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
source /sw/anaconda/3.7-2020.02/thisconda.sh 
# # conda activate /lustre/collider/zhangyulei/DeepLearning/env
conda activate /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/env

GNNPATH=/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn
export PYTHONPATH="${PYTHONPATH}:${GNNPATH}"
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# export HOME="/lustre/collider/luzejia"

# Step -1: Delete training checkpoints
# rm -rf ${GNNPATH}/output.link.rec/model.checkpoints
# rm -rf ${GNNPATH}/apply.link.rec
# rm -rf ${GNNPATH}/output.momentum.rec/model.checkpoints
# rm -rf ${GNNPATH}/apply.momentum.rec.0p999

# Step 0: Data Preparation
Root_file="/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/Train_root_zejia/Merged_Tracker_GNN.root"
# Root_file = "/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/Train_root_copy/Tracker_GNN_200.root"
echo "${Root_file} has been converted."
python3 extra_script/graph_to_disk.py ${Root_file} -o output_single_200 -c 100MB -b -m  #-m for momentum

# Step 1: Link Training
# python3 ${GNNPATH}/run.py DDP link.yaml -w 1 -r --batch_size 64

# Step 2: Link Application
# cp /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output_200.link.rec/model.checkpoints/model_checkpoint_014.pth.tar /lustre/collider/wanghuayang/DeepLearning/Tracking/new_data/checkpoints/luzejia/model_checkpoint_link.pth.tar

python3 ${GNNPATH}/run.py apply /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output -m /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output.link.rec/model.checkpoints/model_checkpoint_014.pth.tar -c link.yaml -o apply.link.rec -p link --batch_size 64
# python3 ${GNNPATH}/run.py apply /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output -m /lustre/collider/luzejia/darkShine/tracking/workspace_test/output.link.rec/model.checkpoints/model_checkpoint_029.pth.tar -c link.yaml -o apply.link.rec -p link --batch_size 32

# Step 3: Momentum Training
python3 ${GNNPATH}/run.py DDP momentum.yaml -w 1 -r --batch_size 64
cp /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output.link.rec/model.checkpoints/model_checkpoint_014.pth.tar /lustre/collider/wanghuayang/DeepLearning/Tracking/new_data/checkpoints/luzejia/model_checkpoint_momentum.pth.tar

# Step 4: Momentum Application
# python3 ${GNNPATH}/run.py apply /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.link.rec/ -m /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output.momentum.rec/model.checkpoints/model_checkpoint_014.pth.tar -c momentum.yaml -o apply.momentum.rec.0p999 -p momentum --batch_size 64
# python3 ${GNNPATH}/run.py apply /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.link.rec/ -m /lustre/collider/luzejia/darkShine/tracking/workspace_test/output.momentum.rec/model.checkpoints/model_checkpoint_014.pth.tar -c momentum.yaml -o apply.momentum.rec.0p999 -p momentum --batch_size 64

# submit job
# condor_submit -interactive interactive.sub

# efficiency calculation
# python3 ${GNNPATH}/read_tt.py 

# exit the interactive shell and the terminal
exit

