import torch
import numpy as np
from utility.DTrack import DTrack
import networkx as nx
from torch_geometric.utils import to_networkx
import os
import re
import time
import glob
import argparse
import torch.multiprocessing as mp
import logging
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Pool

# ---------------------- 辅助函数 ----------------------
def get_file_indices(recon_template, truth_template):
    """动态获取所有 .lt 文件的索引，并确保对应的 .pt 文件存在"""
    recon_files = glob.glob(recon_template.format('*'))
    file_indices = []

    for recon_file in recon_files:
        match = re.search(r'tracks_(\d+)\.lt', recon_file)
        if match:
            idx = int(match.group(1))
            truth_file = truth_template.format(idx)
            if os.path.exists(truth_file):
                file_indices.append(idx)
            else:
                print(f"Warning: No matching .pt file found for {recon_file} (index {idx})")
        else:
            print(f"Warning: Could not extract index from {recon_file}")

    file_indices.sort() 
    print(f"Found {len(file_indices)} valid file pairs")
    return file_indices

def flatten_dtrack_list(nested_list):
    """展平嵌套列表，提取所有DTrack对象"""
    flat_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flat_list.extend(sublist)
        elif isinstance(sublist, DTrack):
            flat_list.append(sublist)
    return flat_list

def count_tracks_in_truth(truth_data):
    """计算真值轨迹数量"""
    G = to_networkx(truth_data, to_undirected=True)
    edges_to_keep = [(edge[0].item(), edge[1].item()) for edge_idx, edge in enumerate(truth_data.edge_index.T) 
                    if truth_data.y[edge_idx].item() == 1]
    G = nx.Graph(edges_to_keep)
    return sum(1 for comp in nx.connected_components(G) if len(comp) >= 3)

# ---------------------- 核心计算函数 ----------------------
def calculate_event_efficiency(event_tracks, truth_data, range_threshold):
    """计算单个事件的效率"""
    truth_edge = 0
    truth_track = count_tracks_in_truth(truth_data)
    good_fit = 0
    good_fit_track = 0
    just_track = 0
    
    # 提取真值边
    truth_edges = []
    for edge_idx, edge in enumerate(truth_data.edge_index.T):
        if truth_data.y[edge_idx].item() == 1:
            truth_edge += 1
            node1_truth = truth_data.x[edge[0]]
            node2_truth = truth_data.x[edge[1]]
            truth_edges.append((node1_truth, node2_truth))
    
    # 遍历轨迹寻找匹配边
    for node1_truth, node2_truth in truth_edges:
        for track in event_tracks:
            if not hasattr(track, 'all_hits') or len(track.all_hits) < 2 or not track.above_four_hits:
                continue
                
            for i in range(len(track.all_hits) - 1):
                node1_recon = track.all_hits[i]
                node2_recon = track.all_hits[i + 1]
                
                if (
                    abs(node1_truth[0].item() - node1_recon[0]) <= range_threshold and
                    abs(node1_truth[2].item() - node1_recon[2]) <= range_threshold and
                    abs(node2_truth[0].item() - node2_recon[0]) <= range_threshold and
                    abs(node2_truth[2].item() - node2_recon[2]) <= range_threshold
                ):
                    good_fit += 1
                    break  # 找到匹配后跳出轨迹循环
    
    # 计算轨迹匹配效率
    for track in event_tracks:
        if not hasattr(track, 'all_hits') or len(track.all_hits) < 2 or not track.above_four_hits:
            continue
            
        just_track += 1
        track_good_fit = True
        
        for i in range(len(track.all_hits) - 1):
            node1_recon = track.all_hits[i]
            node2_recon = track.all_hits[i + 1]
            edge_good_fit = False
            
            for node1_t, node2_t in truth_edges:
                if (
                    abs(node1_t[0].item() - node1_recon[0]) <= range_threshold and
                    abs(node1_t[2].item() - node1_recon[2]) <= range_threshold and
                    abs(node2_t[0].item() - node2_recon[0]) <= range_threshold and
                    abs(node2_t[2].item() - node2_recon[2]) <= range_threshold
                ):
                    edge_good_fit = True
                    break
                    
            if not edge_good_fit:
                track_good_fit = False
                break
                
        if track_good_fit:
            good_fit_track += 1
    
    return {
        'truth_edge': truth_edge,
        'truth_track': truth_track,
        'good_fit': good_fit,
        'good_fit_track': good_fit_track,
        'just_track': just_track
    }

def process_single_file(file_idx, recon_template, truth_template, range_threshold, num_events):
    """处理单个文件的多进程函数"""
    start_time = time.time()
    file_path_recon = recon_template.format(file_idx)
    file_path_truth = truth_template.format(file_idx)
    
    if not os.path.exists(file_path_recon) or not os.path.exists(file_path_truth):
        print(f"[{time.strftime('%H:%M:%S')}] 文件 {file_idx} 不存在，跳过")
        return None
    
    # 加载数据
    data_recon = torch.load(file_path_recon, map_location='cpu')
    data_truth = torch.load(file_path_truth, map_location='cpu')
    events_recon = [flatten_dtrack_list(event) for event in data_recon]
    events_truth = data_truth[:num_events]
    
    # 限制处理事件数量
    num_processed = min(num_events, len(events_recon), len(events_truth))
    event_results = []
    
    for event_id in range(num_processed):
        event_results.append(calculate_event_efficiency(
            events_recon[event_id],
            events_truth[event_id],
            range_threshold
        ))
    
    # 汇总文件级结果
    total_truth_edge = sum(res['truth_edge'] for res in event_results)
    total_truth_track = sum(res['truth_track'] for res in event_results)
    total_good_fit = sum(res['good_fit'] for res in event_results)
    total_good_fit_track = sum(res['good_fit_track'] for res in event_results)
    total_just_track = sum(res['just_track'] for res in event_results)
    
    # 计算文件级效率
    eff_edge = total_good_fit / total_truth_edge if total_truth_edge else 0
    eff_track = total_good_fit_track / total_truth_track if total_truth_track else 0
    eff_find = total_just_track / total_truth_track if total_truth_track else 0
    
    duration = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] 文件 {file_idx} 处理完成 | "
          f"耗时: {duration:.2f}s | "
          f"边效率: {eff_edge:.2%} | "
          f"轨迹效率: {eff_track:.2%}")
    
    return {
        'eff_edge': eff_edge,
        'eff_track': eff_track,
        'eff_find': eff_find,
        'valid_edge': total_truth_edge,
        'valid_track': total_truth_track,
        'valid_find': total_truth_track,
        'events_process': num_processed,
    }

# ---------------------- 并行处理入口 ----------------------
def efficiency_calculation_all(recon_template, truth_template, range_threshold, num_events, file_indices):
    """多进程并行计算所有文件效率"""
    mp.set_start_method('spawn', force=True)  # 避免Windows下的fork问题
    pool = Pool(processes=mp.cpu_count())  # 使用全部CPU核心
    
    results = []
    for file_idx in file_indices:
        results.append(pool.apply_async(process_single_file, 
                                      (file_idx, recon_template, truth_template, range_threshold, num_events)))
    
    pool.close()
    pool.join()
    
    # 汇总结果
    total_eff_edge = 0
    total_eff_track = 0
    total_eff_find = 0
    total_valid_edge = 0
    total_valid_track = 0
    total_valid_find = 0
    total_events_processed = 0
    
    for res in results:
        data = res.get()
        if data is None or data['valid_edge'] == 0:
            continue
        total_eff_edge += data['eff_edge'] * data['valid_edge']
        total_eff_track += data['eff_track'] * data['valid_track']
        total_eff_find += data['eff_find'] * data['valid_find']
        total_valid_edge += data['valid_edge']
        total_valid_track += data['valid_track']
        total_valid_find += data['valid_find']
        total_events_processed += data['events_process']
    
    # 计算全局平均
    avg_eff_edge = total_eff_edge / total_valid_edge if total_valid_edge else 0
    avg_eff_track = total_eff_track / total_valid_track if total_valid_track else 0
    avg_eff_find = total_eff_find / total_valid_find if total_valid_find else 0
    
    print("\n===================== 全局结果 =====================")
    print(f"处理事件总数: {total_events_processed}")
    print(f"边效率: {avg_eff_edge:.2%} ({total_valid_edge} 有效边)")
    print(f"轨迹匹配效率: {avg_eff_track:.2%} ({total_valid_track} 有效轨迹)")
    print(f"轨迹发现效率: {avg_eff_find:.2%} ({total_valid_find} 有效轨迹)")
    print("====================================================")
    
    return avg_eff_edge, avg_eff_track, avg_eff_find

# ---------------------- 主函数（添加argparse参数解析） ----------------------
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='轨迹效率计算程序')
    
    # 添加模板路径前缀参数（核心修改部分）
    parser.add_argument(
        '-recon-prefix', 
        type=str, 
        default='/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/run/single_200_Over/apply.momentum.rec.0p999/DigitizedRecTrk/',
        help='重建轨迹文件的路径前缀（包含文件夹，不包含文件名）'
    )
    parser.add_argument(
        '-truth-prefix', 
        type=str, 
        default='/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/run/single_200_Over/apply.link.rec/DigitizedRecTrk/',
        help='真值轨迹文件的路径前缀（包含文件夹，不包含文件名）'
    )
    
    # 添加其他可配置参数
    parser.add_argument(
        '-range-threshold', 
        type=float, 
        default=0.001, 
        help='距离阈值（默认0.001）'
    )
    parser.add_argument(
        '-num-events', 
        type=int, 
        default=100000, 
        help='每个文件处理的事件数（默认100000）'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 构建完整的模板路径（前缀 + 文件名格式）
    recon_template = os.path.join(args.recon_prefix, 'tracks_{}.lt')  # 拼接前缀和文件名模板
    truth_template = os.path.join(args.truth_prefix, 'graph_{}.pt')   # 拼接前缀和文件名模板
    
    # 获取有效文件索引
    file_indices = get_file_indices(recon_template, truth_template)
    
    # 开始计算
    print(f"检测到 {len(file_indices)} 个有效文件对，开始并行处理...")
    start_time = time.time()
    
    efficiency_calculation_all(
        recon_template=recon_template,
        truth_template=truth_template,
        range_threshold=args.range_threshold,
        num_events=args.num_events,
        file_indices=file_indices
    )
    
    print(f"\n总耗时: {time.time() - start_time:.2f} 秒")