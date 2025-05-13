import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import yaml
from shutil import copyfile
from utility.DTrack import DTrack 
import networkx as nx
from torch_geometric.utils import to_networkx

def flatten_dtrack_list(nested_list):
    """
    将嵌套的列表展平，提取所有 DTrack 对象。
    :param nested_list: 嵌套的列表，包含 DTrack 对象。
    :return: 展平后的 DTrack 对象列表。
    """
    flat_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flat_list.extend(sublist)  # 展平子列表
        elif isinstance(sublist, DTrack):
            flat_list.append(sublist)  # 如果是 DTrack 对象，直接添加
    return flat_list

def count_tracks_in_truth(truth_data):
    """
    计算一个 truth_data 图中有多少条满足首尾相连条件的轨迹。
    :param truth_data: 包含 edge_index 和 y 的图数据。
    :return: 满足条件的轨迹数量。
    """
    # 将 truth_data 转换为 NetworkX 图
    G = to_networkx(truth_data, to_undirected=True)

    # 只保留 y=1 的边
    edges_to_keep = []
    for edge_idx, edge in enumerate(truth_data.edge_index.T):
        if truth_data.y[edge_idx].item() == 1:  # 只保留 y=1 的边
            edges_to_keep.append((edge[0].item(), edge[1].item()))
    G = nx.Graph(edges_to_keep)

    # 计算连通分量
    connected_components = list(nx.connected_components(G))

    # 统计满足条件的轨迹数量
    track_count = 0
    for component in connected_components:
        if len(component) >= 3:
            track_count += 1

    return track_count

def save_results_to_txt(threshold_values, first_threshold_values, efficiency_edge, efficiency_track, output_file):
    """
    将结果保存到文本文件中。
    :param threshold_values: 阈值数组 (Threshold)
    :param first_threshold_values: 第一阈值数组 (First Threshold)
    :param efficiency_edge: 边效率数组 (Edge Efficiency)
    :param efficiency_track: 轨迹效率数组 (Track Efficiency)
    :param output_file: 输出文件路径
    """

    # 打开文件并写入数据
    with open(output_file, 'w') as f:
        # 写入表头
        f.write(f"{'Threshold':<12} {'First_Threshold':<15} {'Edge_Efficiency':<15} {'Track_Efficiency':<15}\n")
        f.write("-" * 60 + "\n")

        # 遍历所有组合并写入每一行数据
        for i, threshold in enumerate(threshold_values):
            for j, first_threshold in enumerate(first_threshold_values):
                edge_eff = efficiency_edge[i][j]
                track_eff = efficiency_track[i][j]
                f.write(f"{threshold:<12.4f} {first_threshold:<15.4f} {edge_eff:<15.4f} {track_eff:<15.4f}\n")

    print(f"All results saved to {output_file}")

def efficiency_calculation(file_path_recon, file_path_truth, range_threshold=0.001, num_events=2300):
    """
    计算重建轨迹的效率。
    :param file_path_recon: .lt 文件路径（重建轨迹）。
    :param file_path_truth: .pt 文件路径（真值数据）。
    :param range_threshold: 判断节点是否匹配的范围阈值。
    :param num_events: 要计算的事件数量。
    :return: 边效率和轨迹效率
    """
    # 加载 .lt 文件中的轨迹数据
    data_recon = torch.load(file_path_recon, map_location='cpu')
    analyzed_tracks_list = [flatten_dtrack_list(event) for event in data_recon]
    
    # 加载 .pt 文件中的真值数据
    data_truth = torch.load(file_path_truth, map_location='cpu')

    total_efficiency_edge = 0
    total_efficiency_track = 0  
    valid_events = 0
    valid_events_track = 0  

    for event_id in range(min(num_events, len(analyzed_tracks_list), len(data_truth))):
        event_tracks = analyzed_tracks_list[event_id]
        truth_data = data_truth[event_id]

        truth_edge = 0
        truth_track = 0
        good_fit = 0
        good_fit_track = 0
        truth_edges = []

        truth_track = count_tracks_in_truth(truth_data)
        for edge_idx, edge in enumerate(truth_data.edge_index.T):
            if truth_data.y[edge_idx].item() == 1:
                truth_edge += 1
                node1_truth = truth_data.x[edge[0]]
                node2_truth = truth_data.x[edge[1]]
                truth_edges.append((node1_truth, node2_truth))

        for track in event_tracks:
            if not hasattr(track, 'all_hits') or len(track.all_hits) < 2 or not track.above_four_hits:
                continue

            for i in range(len(track.all_hits)-1):
                node1_recon = track.all_hits[i]
                node2_recon = track.all_hits[i+1]
                for node1_truth, node2_truth in truth_edges:
                    if (abs(node1_truth[0].item() - node1_recon[0]) <= range_threshold and
                        abs(node1_truth[2].item() - node1_recon[2]) <= range_threshold and
                        abs(node2_truth[0].item() - node2_recon[0]) <= range_threshold and
                        abs(node2_truth[2].item() - node2_recon[2]) <= range_threshold):
                        good_fit += 1
                        break

        for track in event_tracks:
            if not hasattr(track, 'all_hits') or len(track.all_hits) < 2 or not track.above_four_hits:
                continue

            track_good_fit = True
            for i in range(len(track.all_hits)-1):
                node1_recon = track.all_hits[i]
                node2_recon = track.all_hits[i+1]
                edge_good = False
                for node1_truth, node2_truth in truth_edges:
                    if (abs(node1_truth[0].item() - node1_recon[0]) <= range_threshold and
                        abs(node1_truth[2].item() - node1_recon[2]) <= range_threshold and
                        abs(node2_truth[0].item() - node2_recon[0]) <= range_threshold and
                        abs(node2_truth[2].item() - node2_recon[2]) <= range_threshold):
                        edge_good = True
                        break
                if not edge_good:
                    track_good_fit = False
                    break
            if track_good_fit:
                good_fit_track += 1

        if truth_edge > 0:
            event_efficiency_edge = good_fit / truth_edge
            if event_efficiency_edge > 1:
                continue
            total_efficiency_edge += event_efficiency_edge
            valid_events += 1

        if truth_track > 0:
            event_efficiency_track = good_fit_track / truth_track
            if event_efficiency_track > 1:
                continue
            total_efficiency_track += event_efficiency_track
            valid_events_track += 1

    avg_edge = total_efficiency_edge / valid_events if valid_events else 0
    avg_track = total_efficiency_track / valid_events_track if valid_events_track else 0
    return avg_edge, avg_track



def main():
    # 参数设置
    threshold_values = np.linspace(0.4, 0.8, 9)  # 主阈值范围
    first_threshold_values = np.linspace(0.5, 0.9, 9)  # 第一阈值范围
    # threshold_values = np.arange(0.8, 0.2, -0.05)  # 从0.9到0.2，步长0.01
    # first_threshold_values = np.arange(0.75, 0.2, -0.05)
    
    efficiency_edge = np.zeros((len(threshold_values), len(first_threshold_values)))
    efficiency_track = np.zeros((len(threshold_values), len(first_threshold_values)))
    
    # 备份原始配置文件
    yaml_file = '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/momentum.yaml'
    backup_yaml_file = yaml_file + '.bak'
    copyfile(yaml_file, backup_yaml_file)
    
    try:
        for i, threshold in enumerate(threshold_values):
            for j, first_threshold in enumerate(first_threshold_values):
                print(f"Processing threshold={threshold}, first_threshold={first_threshold}")
                
                # 修改配置文件
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                config['data']['threshold'] = float(threshold)
                config['data']['first_threshold'] = float(first_threshold)
                with open(yaml_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                # 运行跟踪算法
                command = [
                    'python3', 
                    '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/run.py', 
                    'apply', 
                    '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.link.rec/',
                    '-m', '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output.momentum.rec/model.checkpoints/model_checkpoint_014.pth.tar',
                    '-c', yaml_file,
                    '-o', 'apply.momentum.rec.0p999',
                    '-p', 'momentum'
                ]
                subprocess.run(command, check=True)
                
                # 计算效率
                file_path_recon = '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.momentum.rec.0p999/DigitizedRecTrk/tracks_0.lt'
                file_path_truth = '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/output/DigitizedRecTrk/graph_list.1.out.pt'
                
                avg_edge, avg_track = efficiency_calculation(file_path_recon, file_path_truth, num_events=2300)
                efficiency_edge[i,j] = avg_edge
                efficiency_track[i,j] = avg_track
    
        # 保存结果到文件
        output_file = "efficiency_results.txt"
        save_results_to_txt(threshold_values, first_threshold_values, efficiency_edge, efficiency_track, output_file)


    finally:
        # 恢复原始配置文件
        copyfile(backup_yaml_file, yaml_file)
    
    # # 绘图
    # fig1 = plt.figure(figsize=(8, 6))
    # ax1 = fig1.add_subplot(111, projection='3d')
    # surf1 = ax1.plot_surface(X, Y, efficiency_edge, cmap='viridis')
    # ax1.set_xlabel('Threshold')
    # ax1.set_ylabel('First Threshold')
    # ax1.set_zlabel('Edge Efficiency')
    # ax1.set_title('Edge Efficiency vs Thresholds')
    # fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    # fig1.savefig('fig/efficiency/edge_efficiency.png', dpi=300, bbox_inches='tight')

    # fig2 = plt.figure(figsize=(8, 6))
    # ax2 = fig2.add_subplot(111, projection='3d')
    # surf2 = ax2.plot_surface(X, Y, efficiency_track, cmap='viridis')
    # ax2.set_xlabel('Threshold')
    # ax2.set_ylabel('First Threshold')
    # ax2.set_zlabel('Track Efficiency')
    # ax2.set_title('Track Efficiency vs Thresholds')
    # fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    # fig2.savefig('fig/efficiency/track_efficiency.png', dpi=300, bbox_inches='tight')

    # print("Separate plots saved as 'edge_efficiency_plot.png' and 'track_efficiency_plot.png'")
    
    # # 输出最佳参数
    # max_edge = efficiency_edge.max()
    # max_edge_pos = np.unravel_index(efficiency_edge.argmax(), efficiency_edge.shape)
    # best_threshold_edge = threshold_values[max_edge_pos[0]]
    # best_first_edge = first_threshold_values[max_edge_pos[1]]
    
    # max_track = efficiency_track.max()
    # max_track_pos = np.unravel_index(efficiency_track.argmax(), efficiency_track.shape)
    # best_threshold_track = threshold_values[max_track_pos[0]]
    # best_first_track = first_threshold_values[max_track_pos[1]]
    
    # print(f"Best Edge Efficiency: {max_edge:.2%} at Threshold={best_threshold_edge:.2f}, First Threshold={best_first_edge:.2f}")
    # print(f"Best Track Efficiency: {max_track:.2%} at Threshold={best_threshold_track:.2f}, First Threshold={best_first_track:.2f}")

if __name__ == "__main__":
    main()