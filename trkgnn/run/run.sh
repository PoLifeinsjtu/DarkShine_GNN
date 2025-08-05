#!/bin/bash
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
source /sw/anaconda/3.7-2020.02/thisconda.sh 
# # conda activate /lustre/collider/zhangyulei/DeepLearning/env
conda activate /lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/env

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}" || exit
echo "Current working directory: $(pwd)"

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
# Root_file="/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/Train_root/Merged_Tracker_GNN.root"
Root_file="/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/Train_root/Tracker_GNN_200.root" # Train_root
# Root_file="/lustre/collider/wanghuayang/DeepLearning/Tracking/new_data/Test_root/test_500.root" # Test_root
echo "${Root_file} has been converted."
# python3 ${GNNPATH}/extra_script/graph_to_disk.py ${Root_file} -o 500/output_test -c 100MB -b -m  #-m for momentum

# Step 1: Link Training
# python3 ${GNNPATH}/run.py DDP link.yaml -w 1 -r --batch_size 64

# Step 2: Link Application
# python3 ${GNNPATH}/run.py apply ${SCRIPT_DIR}/output_train -m ${SCRIPT_DIR}/output.link.rec/model.checkpoints/model_checkpoint_014.pth.tar -c link.yaml -o apply.link.rec -p link --batch_size 64

# Step 3: Momentum Training
# python3 ${GNNPATH}/run.py DDP momentum.yaml -w 1 -r --batch_size 64

# Step 4: Link test Application
python3 ${GNNPATH}/run.py apply ${SCRIPT_DIR}/200/output_test -m ${SCRIPT_DIR}/output.link.rec/model.checkpoints/model_checkpoint_014.pth.tar -c link.yaml -o apply.link.rec -p link --batch_size 64

# Step 5: Momentum Application
python3 ${GNNPATH}/run.py apply ${SCRIPT_DIR}/apply.link.rec/ -m ${SCRIPT_DIR}/output.momentum.rec/model.checkpoints/model_checkpoint_014.pth.tar -c momentum.yaml -o apply.momentum.rec.0p999 -p momentum --batch_size 64

# efficiency calculation
python3 efficiency_compare.py -recon-prefix ${SCRIPT_DIR}/apply.momentum.rec.0p999/DigitizedRecTrk -truth-prefix ${SCRIPT_DIR}/apply.link.rec/DigitizedRecTrk

# Get pt file in edge
# python pt_edge_find_1.py ${SCRIPT_DIR}/apply.link.rec/DigitizedRecTrk/ -processes 2
# python pt_edge_find_2.py ${SCRIPT_DIR}/apply.link.rec/DigitizedRecTrk/ -processes 2
# python score_draw.py ${SCRIPT_DIR}/apply.link.rec/DigitizedRecTrk/ -max-events -1

# exit the interactive shell and the terminal
exit

# submit job
# condor_submit -interactive interactive.sub

