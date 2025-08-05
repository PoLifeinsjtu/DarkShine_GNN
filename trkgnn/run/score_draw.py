import sys
# 强制日志实时刷新（适配Condor输出）
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import torch
import os
import argparse
import gc
import psutil
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转为 MB

def process_single_file(args):
    """单个文件的处理函数（供多进程调用）"""
    file_path, range_threshold, max_events = args
    file_name = os.path.basename(file_path)
    try:
        mem_before = get_memory_usage()
        print(f"[进程 {os.getpid()}] 加载文件: {file_name} (内存: {mem_before:.2f} MB)")
        
        # 加载文件并限制事件数
        with torch.no_grad():  # 禁用梯度计算
            data_list = torch.load(file_path, map_location='cpu')
        if max_events > 0 and len(data_list) > max_events:
            data_list = data_list[:max_events]
        
        total_truth_edges = 0
        total_matched_edges = 0
        total_all_edges = 0

        matched_scores = []  # 存储匹配边的分数
        unmatched_scores = []  # 存储未匹配边的分数

        for i, data in enumerate(data_list):
            if i % 10 == 0:
                mem = get_memory_usage()
                print(f"[进程 {os.getpid()}] 处理事件 {i}/{len(data_list)} (内存: {mem:.2f} MB)")
            
            # 提取真值边（y=1的边）
            truth_edges = []
            if hasattr(data, 'edge_index') and hasattr(data, 'y') and hasattr(data, 'x') and data.x is not None:
                mask = data.y == 1
                truth_edges = data.edge_index.T[mask]
                truth_nodes = [(data.x[edge[0]], data.x[edge[1]]) for edge in truth_edges]
                total_truth_edges += len(truth_edges)
            
            # 提取重建边（edge_attr最后一个分数>0.5的边）
            recon_edges = []
            if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr') and hasattr(data, 'x') and data.x is not None:
                # mask = data.edge_attr[:, -1] > 0.5
                # recon_edges = data.edge_index.T[mask]
                # recon_nodes = [(data.x[edge[0]], data.x[edge[1]]) for edge in recon_edges]
                # total_all_edges += len(data.edge_index.T)  # 记录所有边数
                recon_edges = data.edge_index.T
                recon_nodes = [(data.x[edge[0]], data.x[edge[1]]) for edge in recon_edges]
                total_all_edges += len(recon_edges)  # 记录所有边数
            
            # 匹配边（使用张量操作优化）
            matched_edges = 0
            if truth_nodes and recon_nodes:
                # 转换为张量以进行批量比较
                truth_nodes_tensor = torch.stack([torch.stack([n[0], n[1]]) for n in truth_nodes])
                recon_nodes_tensor = torch.stack([torch.stack([n[0], n[1]]) for n in recon_nodes])
                
                for recon_idx, recon_pair in enumerate(recon_nodes_tensor):
                    node1_recon, node2_recon = recon_pair
                    # 计算所有重建边与当前真值边的差异
                    diff_node1_x = torch.abs(truth_nodes_tensor[:, 0, 0] - node1_recon[0])
                    diff_node1_z = torch.abs(truth_nodes_tensor[:, 0, 2] - node1_recon[2])
                    diff_node2_x = torch.abs(truth_nodes_tensor[:, 1, 0] - node2_recon[0])
                    diff_node2_z = torch.abs(truth_nodes_tensor[:, 1, 2] - node2_recon[2])
                    
                    # 检查是否满足阈值
                    matches = (diff_node1_x <= range_threshold) & (diff_node1_z <= range_threshold) & \
                              (diff_node2_x <= range_threshold) & (diff_node2_z <= range_threshold)
                    if matches.any():
                        matched_edges += 1
                        matched_scores.append(data.edge_attr[recon_idx, -1].item())  # 添加匹配边分数
                    else:
                        unmatched_scores.append(data.edge_attr[recon_idx, -1].item())  # 添加未匹配边分数
            
            total_matched_edges += matched_edges
            
            # 释放内存
            del truth_edges, recon_edges, truth_nodes, recon_nodes
            gc.collect()
        
        mem_after = get_memory_usage()
        total_unmatched_edges = total_all_edges - total_matched_edges
        
        print(f"\n===== 完成文件: {file_name} =====")
        print(f"  该文件总边数 (all): {total_all_edges}")
        print(f"  该文件真值边数 (truth): {total_truth_edges}")
        print(f"  该文件匹配边数 (match): {total_matched_edges}")
        print(f"  该文件未匹配边数 (unmatch): {total_unmatched_edges}")
        if total_truth_edges > 0:
            print(f"  该文件边效率: {total_matched_edges / total_truth_edges:.4f}")
        else:
            print(f"  该文件边效率: 无真值边")
        print(f"================================\n")
        
        return total_matched_edges, total_truth_edges, total_unmatched_edges, total_all_edges, matched_scores, unmatched_scores

    except Exception as e:
        print(f"[进程 {os.getpid()}] 处理文件 {file_name} 失败: {e}")
        return 0, 0, 0, 0, [], []

def main(args):
    directory = args.directory
    range_threshold = args.threshold
    max_events = args.max_events if args.max_events > 0 else 0
    processes = min(args.processes, mp.cpu_count() - 1)
    
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return
    
    pt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误：目录 '{directory}' 中未找到 .pt 文件")
        return
    
    print(f"找到 {len(pt_files)} 个 .pt 文件，使用 {processes} 个进程处理...\n")
    
    # 设置多进程共享策略
    mp.set_start_method('spawn', force=True)
    
    # 准备任务参数
    task_args = [(file_path, range_threshold, max_events) for file_path in pt_files]
    
    # 多进程处理并汇总
    total_matched = 0
    total_truth = 0
    total_unmatched = 0
    total_all = 0
    all_matched_scores = []  # 收集所有匹配边分数
    all_unmatched_scores = []  # 收集所有未匹配边分数
    
    with mp.Pool(processes=processes) as pool:
        for result in tqdm(pool.imap(process_single_file, task_args), total=len(pt_files)):
            matched, truth, unmatched, all_edges, matched_scores, unmatched_scores = result
            total_matched += matched
            total_truth += truth
            total_unmatched += unmatched
            total_all += all_edges
            all_matched_scores.extend(matched_scores)  # 合并匹配边分数
            all_unmatched_scores.extend(unmatched_scores)  # 合并未匹配边分数
    
    # 输出最终汇总结果
    print("\n" + "="*50)
    print("所有文件处理完成，汇总结果如下：")
    print("="*50)
    print(f"总边数 (total all): {total_all}")
    print(f"总真值边数 (total truth): {total_truth}")
    print(f"总匹配边数 (total match): {total_matched}")
    print(f"总未匹配边数 (total unmatch): {total_unmatched}")
    if total_truth > 0:
        print(f"总边效率 (total efficiency): {total_matched / total_truth:.4f} ({total_matched / total_truth * 100:.2f}%)")
    else:
        print("总边效率 (total efficiency): 无有效真值边")
    print("="*50)

    # 绘制匹配边分数直方图
    if all_matched_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(all_matched_scores, bins=50, alpha=0.7, color='green')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Matched Edge Scores Distribution')
        plt.grid(alpha=0.3)
        plt.savefig('Matched_Edge_Scores_Distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("没有匹配边分数数据，无法绘制匹配边分数图")

    # 绘制未匹配边分数直方图
    if all_unmatched_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(all_unmatched_scores, bins=50, alpha=0.7, color='red')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Unmatched Edge Scores Distribution')
        plt.grid(alpha=0.3)
        plt.savefig('Unmatched_Edge_Scores_Distribution.png', dpi=300, bbox_inches='tight')
        plt.close
    else:
        print("没有未匹配边分数数据，无法绘制未匹配边分数图")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="边效率计算（每个文件实时输出结果）")
    parser.add_argument("directory", type=str, help="包含pt文件的目录路径")
    parser.add_argument("-threshold", type=float, default=0.001, help="匹配阈值（默认：0.001）")
    parser.add_argument("-max-events", type=int, default=-1, help="每个文件最多处理的事件数（默认：全部）")
    parser.add_argument("-processes", type=int, default=2, help="并行进程数（默认：2）")
    args = parser.parse_args()
    main(args)