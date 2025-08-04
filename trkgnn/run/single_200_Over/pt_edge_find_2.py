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
from tqdm import tqdm
import csv

def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转为 MB

def calculate_metrics(tp, fn, fp, tn):
    """计算 Precision, Recall, Specificity, Accuracy, F1-Score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, specificity, accuracy, f1_score

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
        
        total_all_edges = 0
        total_matched_edges = 0
        total_truth_edges = 0
        tp = 0
        fn = 0
        fp = 0
        tn = 0

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
            
            # 提取所有边
            all_edges = 0
            if hasattr(data, 'edge_index'):
                all_edges = data.edge_index.shape[1]
                total_all_edges += all_edges
            
            # 提取重建边（edge_attr最后一个分数>0.5的边）
            recon_edges = []
            recon_scores = []
            if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr') and hasattr(data, 'x') and data.x is not None:
                mask = data.edge_attr[:, -1] > 0
                recon_edges = data.edge_index.T[mask]
                recon_nodes = [(data.x[edge[0]], data.x[edge[1]]) for edge in recon_edges]
                recon_scores = data.edge_attr[:, -1][mask]
            
            # 匹配边（使用张量操作优化）
            matched_edges = 0
            matched_indices = []
            if truth_nodes and recon_nodes:
                truth_nodes_tensor = torch.stack([torch.stack([n[0], n[1]]) for n in truth_nodes])
                recon_nodes_tensor = torch.stack([torch.stack([n[0], n[1]]) for n in recon_nodes])
                
                for truth_idx, truth_pair in enumerate(truth_nodes_tensor):
                    node1_truth, node2_truth = truth_pair
                    diff_node1_x = torch.abs(recon_nodes_tensor[:, 0, 0] - node1_truth[0])
                    diff_node1_z = torch.abs(recon_nodes_tensor[:, 0, 2] - node1_truth[2])
                    diff_node2_x = torch.abs(recon_nodes_tensor[:, 1, 0] - node2_truth[0])
                    diff_node2_z = torch.abs(recon_nodes_tensor[:, 1, 2] - node2_truth[2])
                    
                    matches = (diff_node1_x <= range_threshold) & (diff_node1_z <= range_threshold) & \
                              (diff_node2_x <= range_threshold) & (diff_node2_z <= range_threshold)
                    if matches.any():
                        # matched_idx = matches.nonzero(as_tuple=True)[0][0].item()
                        matched_edges += 1
                        # matched_indices.append(matched_idx)

                for recon_idx, recon_pair in enumerate(recon_nodes_tensor):
                    node1_recon, node2_recon = recon_pair
                    diff_node1_x = torch.abs(truth_nodes_tensor[:, 0, 0] - node1_recon[0])
                    diff_node1_z = torch.abs(truth_nodes_tensor[:, 0, 2] - node1_recon[2])
                    diff_node2_x = torch.abs(truth_nodes_tensor[:, 1, 0] - node2_recon[0])
                    diff_node2_z = torch.abs(truth_nodes_tensor[:, 1, 2] - node2_recon[2])
                    
                    matches = (diff_node1_x <= range_threshold) & (diff_node1_z <= range_threshold) & \
                              (diff_node2_x <= range_threshold) & (diff_node2_z <= range_threshold)
                    if matches.any():
                        matched_idx = matches.nonzero(as_tuple=True)[0][0].item()
                        # matched_edges += 1
                        matched_indices.append(matched_idx)
            
            total_matched_edges += matched_edges
            
            # 计算 TP, FN, FP, TN
            if recon_scores is not None:
                for idx, score in enumerate(recon_scores):
                    if idx in matched_indices:
                        if score >= 0.5:
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if score >= 0.5:
                            fp += 1
                        else:
                            fn += 1
            
            # 释放内存
            del truth_edges, recon_edges, truth_nodes, recon_nodes
            gc.collect()
        
        mem_after = get_memory_usage()
        
        print(f"\n===== 完成文件: {file_name} =====")
        print(f"  该文件总边数 (all): {total_all_edges}")
        print(f"  该文件真值边数 (truth): {total_truth_edges}")
        print(f"  该文件匹配边数 (match): {total_matched_edges}")
        print(f"  该文件TP: {tp}")
        print(f"  该文件FN: {fn}")
        print(f"  该文件FP: {fp}")
        print(f"  该文件TN: {tn}")
        if total_truth_edges > 0:
            print(f"  该文件边效率: {total_matched_edges / total_truth_edges:.4f}")
        else:
            print(f"  该文件边效率: 无真值边")
        print(f"================================\n")
        
        return total_matched_edges, total_truth_edges, tp, fn, fp, tn, file_name, total_all_edges
    
    except Exception as e:
        print(f"[进程 {os.getpid()}] 处理文件 {file_name} 失败: {e}")
        return 0, 0, 0, 0, 0, 0, file_name, 0

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
    
    # 打开 CSV 文件
    with open('results_2.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file name', 'all_edges', 'TP', 'FN', 'FP', 'TN', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-Score'])  # 写入表头
        
        # 准备任务参数
        task_args = [(file_path, range_threshold, max_events) for file_path in pt_files]
        
        # 多进程处理并汇总
        total_matched = 0
        total_truth = 0
        total_tp = 0
        total_fn = 0
        total_fp = 0
        total_tn = 0
        file_results = []
        
        with mp.Pool(processes=processes) as pool:
            for matched, truth, tp, fn, fp, tn, file_name, all_edges in tqdm(pool.imap(process_single_file, task_args), total=len(pt_files)):
                total_matched += matched
                total_truth += truth
                total_tp += tp
                total_fn += fn
                total_fp += fp
                total_tn += tn
                file_results.append((file_name, matched, truth, tp, fn, fp, tn, all_edges))
                
                # 计算指标
                precision, recall, specificity, accuracy, f1_score = calculate_metrics(tp, fn, fp, tn)
                
                # 写入 CSV
                csv_writer.writerow([file_name, all_edges, tp, fn, fp, tn, precision, recall, specificity, accuracy, f1_score])
        
        total_all_edges = sum(r[7] for r in file_results)  # 总边数总和
        # 计算总体指标（基于总TP/总FN/总FP/总TN）
        all_precision, all_recall, all_specificity, all_accuracy, all_f1 = calculate_metrics(
            total_tp, total_fn, total_fp, total_tn
        )

        # 将"all"行写入CSV
        csv_writer.writerow([
            "all",  # 文件名设为"all"
            total_all_edges,  # 总边数总和
            total_tp,  # 总TP
            total_fn,  # 总FN
            total_fp,  # 总FP
            total_tn,  # 总TN
            all_precision,  # 总体precision
            all_recall,  # 总体recall
            all_specificity,  # 总体specificity
            all_accuracy,  # 总体accuracy
            all_f1  # 总体f1-score
        ])

        # 输出最终汇总结果
        print("\n" + "="*50)
        print("所有文件处理完成，汇总结果如下：")
        print("="*50)
        print(f"总边数 (total all): {sum(r[7] for r in file_results)}")
        print(f"总真值边数 (total truth): {total_truth}")
        print(f"总匹配边数 (total match): {total_matched}")
        print(f"总TP: {total_tp}")
        print(f"总FN: {total_fn}")
        print(f"总FP: {total_fp}")
        print(f"总TN: {total_tn}")
        avg_precision = sum(calculate_metrics(r[3], r[4], r[5], r[6])[0] for r in file_results) / len(pt_files) if pt_files else 0.0
        avg_recall = sum(calculate_metrics(r[3], r[4], r[5], r[6])[1] for r in file_results) / len(pt_files) if pt_files else 0.0
        avg_specificity = sum(calculate_metrics(r[3], r[4], r[5], r[6])[2] for r in file_results) / len(pt_files) if pt_files else 0.0
        avg_accuracy = sum(calculate_metrics(r[3], r[4], r[5], r[6])[3] for r in file_results) / len(pt_files) if pt_files else 0.0
        avg_f1_score = sum(calculate_metrics(r[3], r[4], r[5], r[6])[4] for r in file_results) / len(pt_files) if pt_files else 0.0
        print(f"平均Precision: {avg_precision:.4f}")
        print(f"平均Recall: {avg_recall:.4f}")
        print(f"平均Specificity: {avg_specificity:.4f}")
        print(f"平均Accuracy: {avg_accuracy:.4f}")
        print(f"平均F1-Score: {avg_f1_score:.4f}")
        if total_truth > 0:
            print(f"总边效率 (total efficiency): {total_matched / total_truth:.4f} ({total_matched / total_truth * 100:.2f}%)")
        else:
            print("总边效率 (total efficiency): 无有效真值边")
        print("="*50)
        print("结果已写入 results_2.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="边效率计算（每个文件实时输出结果并写入CSV）")
    parser.add_argument("directory", type=str, help="包含pt文件的目录路径")
    parser.add_argument("-threshold", type=float, default=0.001, help="匹配阈值（默认：0.001）")
    parser.add_argument("-max-events", type=int, default=-1, help="每个文件最多处理的事件数（默认：100）")
    parser.add_argument("-processes", type=int, default=2, help="并行进程数（默认：2）")
    args = parser.parse_args()
    main(args)