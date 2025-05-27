import os
import torch

# output_dir = "/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output/DigitizedRecTrk"
output_dir = "/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.link.rec/DigitizedRecTrk"
total_graphs = 0

for file in sorted(os.listdir(output_dir)):
    # if file.startswith("graph_list.") and file.endswith(".out.pt"):
    if file.startswith("graph_") and file.endswith(".pt"):
        path = os.path.join(output_dir, file)
        graphs = torch.load(path)
        total_graphs += len(graphs)
        print(f"{file}: {len(graphs)} graphs")

print(f"Total graphs saved by LinkNet: {total_graphs}")