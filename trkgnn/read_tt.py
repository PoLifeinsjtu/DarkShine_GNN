import torch
import matplotlib.pyplot as plt
import numpy as np
from utility.DTrack import DTrack
import networkx as nx
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import os


def extract_dtrack_data(data):
    """
    从嵌套数据结构中提取 DTrack 对象的 hits 和 E/E0 信息。
    :param data: 嵌套的列表或字典，包含 DTrack 对象。
    :return: 提取的 hits 和 E/E0 数据列表。
    """
    hits = []
    energies = []

    def recursive_extract(item):
        if isinstance(item, list):
            for sub_item in item:
                recursive_extract(sub_item)
        elif isinstance(item, DTrack):
            hits.append(item.no_hits)
            # 假设 DTrack 对象有 energy 和 energy_initial 属性
            if hasattr(item, 'energy') and hasattr(item, 'energy_initial') and item.energy_initial != 0:
                energies.append(item.energy / item.energy_initial)  # 计算 E/E0
            else:
                energies.append(None)  # 如果无法计算，添加占位符
        elif isinstance(item, dict):
            for value in item.values():
                recursive_extract(value)

    recursive_extract(data)
    return hits, energies



def plot_tracks_2d(analyzed_tracks_list, num_tracks):
    """
    绘制前几条轨迹的所有 hits 的 x-z 平面图，并使用 E/E0 作为每条轨迹的标签。
    :param analyzed_tracks_list: 包含 DTrack 对象的列表。
    :param num_tracks: 要绘制的轨迹数量。
    """
    plt.figure(figsize=(10, 8))
    
    for idx, track in enumerate(analyzed_tracks_list[:num_tracks]):
        if not isinstance(track, DTrack):
            print(f"Skipping non-DTrack object at index {idx}: {type(track)}")
            continue
        
        # 检查 all_hits 是否存在并包含数据
        if hasattr(track, 'all_hits') and len(track.all_hits) > 0:
            # 提取所有 hits 的 x 和 z 坐标
            x_coords = [hit[0] for hit in track.all_hits]
            z_coords = [hit[2] for hit in track.all_hits]
            
            # 计算并添加 p_track 标签
            e_over_e0 = getattr(track, 'p_avg', None)
            label = f'p_track: {e_over_e0ß:.4f}' if e_over_e0 is not None else f'Track {idx + 1}'

            # 打印调试信息
            # print(f"Track {idx + 1}: all_hits = {track.all_hits}")
            
            # 绘制轨迹
            # plt.plot(z_coords, x_coords, marker='o')
            plt.plot(z_coords, x_coords, marker='o', label=label)

        else:
            print(f"Track {idx + 1} has missing or empty hit data.")
        
        if hasattr(track, 'vertex_hit'):
            print(f"yes")
    
    # 添加图例和标签
    plt.title("Tracks in x-z Plane")
    plt.xlabel("z (mm)")
    plt.ylabel("x (mm)")
    if any(line.get_label() for line in plt.gca().get_lines()):
        plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('tracks_2d.png')



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


def load_and_plot_tracks(file_path, num_tracks):
    """
    加载 .lt 文件并绘制前几条轨迹的 x-z 平面图。
    :param file_path: .lt 文件路径。
    :param num_tracks: 要绘制的轨迹数量。
    """
    # 加载 .lt 文件中的轨迹数据
    data = torch.load(file_path, map_location='cpu')
    print(f"Loaded data type: {type(data)}")
    
    # 展平嵌套列表
    analyzed_tracks_list = flatten_dtrack_list(data)
    # print(f"Extracted {len(analyzed_tracks_list)} DTrack objects.")

    # 绘制轨迹
    plot_tracks_2d(analyzed_tracks_list, num_tracks=num_tracks)

def extract_initial_final_momentum(dtrack_list):
    """
    从 DTrack 对象列表中提取初始动量 (p_i) 和最终动量 (p_f)。
    :param dtrack_list: 包含 DTrack 对象的列表。
    :return: 包含 (p_i, p_f) 的列表。
    """
    momentum_data = []

    for idx, track in enumerate(dtrack_list):
        if not isinstance(track, DTrack):
            print(f"Skipping non-DTrack object at index {idx}: {type(track)}")
            continue

        # 检查是否有 p_i 和 p_f 属性
        if hasattr(track, 'p_i') and hasattr(track, 'p_f'):
            momentum_data.append((track.p_i, track.p_f))
        else:
            print(f"Track {idx} does not have p_i or p_f attributes.")

    return momentum_data

def inspect_file_structure(file_path):
    data = torch.load(file_path, map_location='cpu')
    print(f"Loaded data type: {type(data)}")
    if isinstance(data, list):
        print(f"First element type: {type(data[0])}")
        print(f"First element length: {len(data[0])}")
        if isinstance(data[0], list):
            print(f"Nested list first element type: {type(data[0][0])}")
            # print(f"Nested list first element length: {len(data[0][0])}") 
    elif isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        for key, value in data.items():
            print(f"Key: {key}, Value type: {type(value)}")
    print(data[:5])

    # if isinstance(data, list) and all(hasattr(d, 'p') for d in data[:5]):
    #     for i, d in enumerate(data[:5]):
    #         print(f"Data {i} p: {d.p}")

    analyzed_tracks_list = [flatten_dtrack_list(event) for event in data]
    for event_id, event_tracks in enumerate(analyzed_tracks_list[:5]):
        print(f"Event {event_id}:")
        for idx, track in enumerate(event_tracks):
            if not isinstance(track, DTrack):
                print(f"Skipping non-DTrack object at index {idx}: {type(track)}")
                continue

            if hasattr(track, 'p_all'):
                # 检查 p_all 是否存在并包含数据
                if len(track.p_all) > 0:
                    print(f"Track {idx} p_all: {track.p_all}")
                


def plot_tracks_2d_by_event(analyzed_tracks_list, num_events):
    """
    绘制每个事件的所有轨迹的 x-z 平面图，并使用 E/E0 作为每条轨迹的标签。
    :param analyzed_tracks_list: 嵌套的列表，每个子列表代表一个事件，包含 DTrack 对象。
    :param num_events: 要绘制的事件数量。
    """
    for event_id, event_tracks in enumerate(analyzed_tracks_list[:num_events]):
        plt.figure(figsize=(10, 8))
        
        for idx, track in enumerate(event_tracks):
            if not isinstance(track, DTrack):
                print(f"Skipping non-DTrack object in event {event_id}, index {idx}: {type(track)}")
                continue
            
            # 检查 all_hits 是否存在并包含数据
            if hasattr(track, 'all_hits') and len(track.all_hits) > 0 and track.above_four_hits:
                # 提取所有 hits 的 x 和 z 坐标
                x_coords = [hit[0] for hit in track.all_hits]
                z_coords = [hit[2] for hit in track.all_hits]
                
                # 计算并添加 p_track 标签
                e_over_e0 = getattr(track, 'p_avg', None)
                label = f'p_track: {e_over_e0:.4f}' if e_over_e0 is not None else f'Track {idx + 1}'

                # 绘制轨迹
                plt.plot(z_coords, x_coords, marker='o', label=label, color='blue')

            else:
                print(f"Track {idx + 1} in event {event_id} has missing or empty hit data.")
        
        # 添加图例和标签
        plt.title(f"Event {event_id} - Tracks in x-z Plane")
        plt.xlabel("z (mm)")
        plt.ylabel("x (mm)")
        if any(line.get_label() for line in plt.gca().get_lines()):
            plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.savefig(f'fig/recon_tracks/event_{event_id}.png')
        plt.close()


def load_and_plot_recon_tracks_by_event(file_path_recon, file_path_raw, raw_hits, num_events):
    """
    加载 .lt 文件和 .pt 文件并绘制每个事件的轨迹的 x-z 平面图。
    :param file_path_recon: .lt 文件路径（重建轨迹）。
    :param file_path_raw: .pt 文件路径（原始数据）。
    :param raw_hits: 是否加载原始 hits。
    :param num_events: 要绘制的事件数量。
    """
    # 加载 .lt 文件中的轨迹数据
    data_recon = torch.load(file_path_recon, map_location='cpu')
    # print(f"Loaded .lt data type: {type(data_recon)}")
    
    # 展平嵌套列表，每个子列表代表一个事件
    analyzed_tracks_list = [flatten_dtrack_list(event) for event in data_recon]
    # print(f"Extracted {len(analyzed_tracks_list)} events from .lt file.")

    # 如果 raw_hits 为 True，则加载 .pt 文件
    if raw_hits:
        data_raw = torch.load(file_path_raw, map_location='cpu')
        # print(f"Loaded .pt data type: {type(data_raw)}")
        
        # 遍历每个事件，绘制 .lt 和 .pt 数据
        for event_id in range(min(num_events, len(analyzed_tracks_list))):
            plt.figure(figsize=(10, 8))
            
            # 绘制 .lt 文件的轨迹
            event_tracks = analyzed_tracks_list[event_id]
            for idx, track in enumerate(event_tracks):
                if not isinstance(track, DTrack):
                    print(f"Skipping non-DTrack object in event {event_id}, index {idx}: {type(track)}")
                    continue
                
                # 检查 all_hits 是否存在并包含数据
                if hasattr(track, 'all_hits') and len(track.all_hits) > 0 and track.above_four_hits:
                # if hasattr(track, 'all_hits') and len(track.all_hits) > 0:
                    # 提取所有 hits 的 x 和 z 坐标
                    x_coords = [hit[0] for hit in track.all_hits]
                    z_coords = [hit[2] for hit in track.all_hits]
                    
                    # 计算并添加 p_track 标签
                    e_over_e0 = getattr(track, 'p_avg', None)
                    label = f'p_track: {e_over_e0:.4f}' if e_over_e0 is not None else f'Track {idx + 1}'

                    # 绘制轨迹，所有线为蓝色
                    plt.plot(z_coords, x_coords, marker='o', label=label, color='blue')

            # 绘制 .pt 文件的图数据
            if event_id < len(data_raw):
                data = data_raw[event_id]
                G = to_networkx(data, to_undirected=True)  # 转换为 NetworkX 图

                # 获取节点的坐标信息
                if hasattr(data, 'x') and data.x is not None:
                    pos = {i: (data.x[i][0].item(), data.x[i][2].item()) for i in range(data.num_nodes)}
                else:
                    # pos = nx.spring_layout(G, dim=2)  # 如果没有坐标信息，使用 spring 布局
                    continue

                # 绘制节点（空心）
                # for node, (x, z) in pos.items():
                #     plt.scatter(z, x, s=100, facecolors='none', edgecolors='k', alpha=0.7)

                # 绘制边（raw hits画法）
                # for edge in G.edges():
                #     x = [pos[edge[0]][0], pos[edge[1]][0]]
                #     z = [pos[edge[0]][1], pos[edge[1]][1]]

                #     # 获取边的属性并计算透明度
                #     edge_index = list(G.edges()).index(edge)
                #     if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                #         score = data.edge_attr[edge_index][3].item()  # 获取最后一个数作为 score
                #         alpha = max(0, min(score, 1))  # 限制 alpha 在 [0, 1] 范围内
                #     else:
                #         alpha = 0  # 默认透明度

                #     plt.plot(z, x, c='black', alpha=alpha, linestyle='--')  # 使用虚线


                # 绘制边（truth 画法）
                for edge_idx, edge in enumerate(G.edges()):
                    # 检查边的 truth 值 (y)
                    if data.y[edge_idx].item() == 1:  # 只绘制 y=1 的边
                        x = [pos[edge[0]][0], pos[edge[1]][0]]
                        z = [pos[edge[0]][1], pos[edge[1]][1]]

                        # 绘制边
                        plt.plot(z, x, c='black', linestyle='--')  # 使用虚线

                        # 绘制端点（空心圆圈）
                        plt.scatter(z[0], x[0], s=120, facecolors='none', edgecolors='black', alpha=0.7)
                        plt.scatter(z[1], x[1], s=120, facecolors='none', edgecolors='black', alpha=0.7)


            # 添加图例和标签
            plt.title(f"Event {event_id} - Tracks in x-z Plane")
            plt.xlabel("z (mm)")
            plt.ylabel("x (mm)")
            plt.legend()
            plt.grid(True)

            # 保存图形
            output_path = f"fig/truth_recon/compare/event_{event_id}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Event {event_id} saved to {output_path}")
    else:
        # 如果 raw_hits 为 False，仅绘制 .lt 文件的轨迹
        plot_tracks_2d_by_event(analyzed_tracks_list, num_events=num_events)

def load_and_plot_yz(file_path_recon, file_path_truth, raw_hits, num_events):
    # 加载 .lt 文件中的轨迹数据
    data_recon = torch.load(file_path_recon, map_location='cpu')
    # print(f"Loaded .lt data type: {type(data_recon)}")
    
    # 展平嵌套列表，每个子列表代表一个事件
    analyzed_tracks_list = [flatten_dtrack_list(event) for event in data_recon]
    # print(f"Extracted {len(analyzed_tracks_list)} events from .lt file.")

    # 如果 raw_hits 为 True，则加载 .pt 文件
    if raw_hits:
        data_raw = torch.load(file_path_raw, map_location='cpu')
        print(f"Loaded .pt data type: {type(data_raw)}")
        
        # 遍历每个事件，绘制 .lt 和 .pt 数据
        for event_id in range(min(num_events, len(analyzed_tracks_list))):
            plt.figure(figsize=(10, 8))
            
            # 绘制 .lt 文件的轨迹
            event_tracks = analyzed_tracks_list[event_id]
            for idx, track in enumerate(event_tracks):
                if not isinstance(track, DTrack):
                    print(f"Skipping non-DTrack object in event {event_id}, index {idx}: {type(track)}")
                    continue
                
                # 检查 all_hits 是否存在并包含数据
                if hasattr(track, 'all_hits') and len(track.all_hits) > 0 and track.above_four_hits:
                # if hasattr(track, 'all_hits') and len(track.all_hits) > 0:
                    # 提取所有 hits 的 y 和 z 坐标
                    y_coords = [hit[1] for hit in track.all_hits]
                    z_coords = [hit[2] for hit in track.all_hits]
                    
                    # 计算并添加 p_track 标签
                    e_over_e0 = getattr(track, 'p_avg', None)
                    label = f'p_track: {e_over_e0:.4f}' if e_over_e0 is not None else f'Track {idx + 1}'

                    # 绘制轨迹，所有线为蓝色
                    plt.plot(z_coords, y_coords, marker='o', label=label, color='blue')

            # 绘制 .pt 文件的图数据
            if event_id < len(data_raw):
                data = data_raw[event_id]
                G = to_networkx(data, to_undirected=True)  # 转换为 NetworkX 图

                # 获取节点的坐标信息
                if hasattr(data, 'x') and data.x is not None:
                    pos = {i: (data.x[i][1].item(), data.x[i][2].item()) for i in range(data.num_nodes)}
                else:
                    # pos = nx.spring_layout(G, dim=2)  # 如果没有坐标信息，使用 spring 布局
                    continue

                # 绘制节点（空心）
                # for node, (x, z) in pos.items():
                #     plt.scatter(z, x, s=100, facecolors='none', edgecolors='k', alpha=0.7)

                # 绘制边（raw hits画法）
                # for edge in G.edges():
                #     x = [pos[edge[0]][0], pos[edge[1]][0]]
                #     z = [pos[edge[0]][1], pos[edge[1]][1]]

                #     # 获取边的属性并计算透明度
                #     edge_index = list(G.edges()).index(edge)
                #     if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                #         score = data.edge_attr[edge_index][3].item()  # 获取最后一个数作为 score
                #         alpha = max(0, min(score, 1))  # 限制 alpha 在 [0, 1] 范围内
                #     else:
                #         alpha = 0  # 默认透明度

                #     plt.plot(z, x, c='black', alpha=alpha, linestyle='--')  # 使用虚线


                # 绘制边（truth 画法）
                for edge_idx, edge in enumerate(G.edges()):
                    # 检查边的 truth 值 (y)
                    if data.y[edge_idx].item() == 1:  # 只绘制 y=1 的边
                        y = [pos[edge[0]][0], pos[edge[1]][0]]
                        z = [pos[edge[0]][1], pos[edge[1]][1]]

                        # 绘制边
                        plt.plot(z, y, c='black', linestyle='--')  # 使用虚线

                        # 绘制端点（空心圆圈）
                        plt.scatter(z[0], y[0], s=100, facecolors='none', edgecolors='black', alpha=0.7)
                        plt.scatter(z[1], y[1], s=100, facecolors='none', edgecolors='black', alpha=0.7)


            # 添加图例和标签
            plt.title(f"Event {event_id} - Tracks in y-z Plane")
            plt.xlabel("z (mm)")
            plt.ylabel("y (mm)")
            plt.legend()
            plt.grid(True)

            # 保存图形
            output_path = f"fig/truth_recon/y-z/event_{event_id}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Event {event_id} saved to {output_path}")
    else:
        # 如果 raw_hits 为 False，仅绘制 .lt 文件的轨迹
        plot_tracks_2d_by_event(analyzed_tracks_list, num_events=num_events)


def threeD_plot(file_path_recon, file_path_truth, raw_hits, num_events):
    # 加载 .lt 文件中的轨迹数据
    data_recon = torch.load(file_path_recon, map_location='cpu')
    
    # 展平嵌套列表，每个子列表代表一个事件
    analyzed_tracks_list = [flatten_dtrack_list(event) for event in data_recon]

    # 如果 raw_hits 为 True，则加载 .pt 文件
    if raw_hits:
        data_raw = torch.load(file_path_truth, map_location='cpu')
        print(f"Loaded .pt data type: {type(data_raw)}")
        
        # 遍历每个事件，绘制 .lt 和 .pt 数据
        for event_id in range(min(num_events, len(analyzed_tracks_list))):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')  # 创建 3D 子图
            
            # 绘制 .lt 文件的轨迹
            event_tracks = analyzed_tracks_list[event_id]
            for idx, track in enumerate(event_tracks):
                if not isinstance(track, DTrack):
                    print(f"Skipping non-DTrack object in event {event_id}, index {idx}: {type(track)}")
                    continue
                
                # 检查 all_hits 是否存在并包含数据
                if hasattr(track, 'all_hits') and len(track.all_hits) > 0 and track.above_four_hits:
                    # 提取所有 hits 的 x, y 和 z 坐标
                    x_coords = [hit[0] for hit in track.all_hits]
                    y_coords = [hit[1] for hit in track.all_hits]
                    z_coords = [hit[2] for hit in track.all_hits]
                    
                    # 计算并添加 p_track 标签
                    e_over_e0 = getattr(track, 'p_avg', None)
                    label = f'p_track: {e_over_e0:.4f}' if e_over_e0 is not None else f'Track {idx + 1}'

                    # 绘制轨迹
                    ax.plot3D(x_coords, y_coords, z_coords, marker='o', label=label, color='blue')

            # 绘制 .pt 文件的图数据
            if event_id < len(data_raw):
                data = data_raw[event_id]
                G = to_networkx(data, to_undirected=True)  # 转换为 NetworkX 图

                # 获取节点的坐标信息
                if hasattr(data, 'x') and data.x is not None:
                    pos = {i: (data.x[i][0].item(), data.x[i][1].item(), data.x[i][2].item()) for i in range(data.num_nodes)}
                else:
                    continue

                # 绘制边（truth 画法）
                for edge_idx, edge in enumerate(G.edges()):
                    # 检查边的 truth 值 (y)
                    if data.y[edge_idx].item() == 1:  # 只绘制 y=1 的边
                        x = [pos[edge[0]][0], pos[edge[1]][0]]
                        y = [pos[edge[0]][1], pos[edge[1]][1]]
                        z = [pos[edge[0]][2], pos[edge[1]][2]]

                        # 绘制边
                        ax.plot3D(x, y, z, c='black', linestyle='--')  # 使用虚线

                        # 绘制端点（空心圆圈）
                        ax.scatter3D(x[0], y[0], z[0], s=100, facecolors='none', edgecolors='black', alpha=0.7)
                        ax.scatter3D(x[1], y[1], z[1], s=100, facecolors='none', edgecolors='black', alpha=0.7)

            # 添加图例和标签
            ax.set_title(f"Event {event_id} - Tracks in 3D Plane")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("z (mm)")
            ax.legend()
            ax.grid(True)

            # 保存图形
            output_path = f"fig/truth_recon/3D/event_{event_id}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Event {event_id} saved to {output_path}")
    else:
        print("Raw hits not provided, skipping .pt file plotting.")

def print_position(file_path_recon, file_path_truth, raw_hits, num_events):
    # 加载 .lt 文件中的轨迹数据
    data_recon = torch.load(file_path_recon, map_location='cpu')
    
    # 展平嵌套列表，每个子列表代表一个事件
    analyzed_tracks_list = [flatten_dtrack_list(event) for event in data_recon]

    # 如果 raw_hits 为 True，则加载 .pt 文件
    if raw_hits:
        data_raw = torch.load(file_path_truth, map_location='cpu')
        print(f"Loaded .pt data type: {type(data_raw)}")
        
        # 遍历每个事件，打印 .lt 和 .pt 数据
        # for event_id in range(min(num_events, len(analyzed_tracks_list))):
        event_id = num_events
        if event_id < len(analyzed_tracks_list):
            print(f"Event {event_id}:")

            # 打印 .lt 文件的轨迹
            event_tracks = analyzed_tracks_list[event_id]
            for idx, track in enumerate(event_tracks):
                if not isinstance(track, DTrack):
                    print(f"Skipping non-DTrack object in event {event_id}, index {idx}: {type(track)}")
                    continue
                
                # 检查 all_hits 是否存在并包含数据
                if hasattr(track, 'all_hits') and len(track.all_hits) > 0 and track.above_four_hits:
                    # 提取所有 hits 的 x, y 和 z 坐标
                        x_coords = [hit[0] for hit in track.all_hits]
                        y_coords = [hit[1] for hit in track.all_hits]
                        z_coords = [hit[2] for hit in track.all_hits]
                        
                        # 打印轨迹坐标
                        for i in range(len(x_coords)):
                            print(f"Track {idx + 1} Hit {i}: (x, y, z): ({x_coords[i]}, {y_coords[i]}, {z_coords[i]})\n")

            # 打印 .pt 文件的图数据
            if event_id < len(data_raw):
                data = data_raw[event_id]
                G = to_networkx(data, to_undirected=True)  # 转换为 NetworkX 图

                # 获取节点的坐标信息
                if hasattr(data, 'x') and data.x is not None:
                    pos = {i: (data.x[i][0].item(), data.x[i][1].item(), data.x[i][2].item()) for i in range(data.num_nodes)}
                # else:
                #     continue
 

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

def efficiency_calculation(file_path_recon, file_path_truth, range_threshold, num_events):
    """
    计算重建轨迹的效率。
    :param file_path_recon: .lt 文件路径（重建轨迹）。
    :param file_path_truth: .pt 文件路径（真值数据）。
    :param range_threshold: 判断节点是否匹配的范围阈值。
    :param num_events: 要计算的事件数量。
    """
    # 加载 .lt 文件中的轨迹数据
    data_recon = torch.load(file_path_recon, map_location='cpu')
    print(f"Loaded .lt data type: {type(data_recon)}")
    
    # 展平嵌套列表，每个子列表代表一个事件
    analyzed_tracks_list = [flatten_dtrack_list(event) for event in data_recon]
    print(f"Loaded .lt file {len(analyzed_tracks_list)} events.")

    # 加载 .pt 文件中的真值数据
    data_truth = torch.load(file_path_truth, map_location='cpu')
    print(f"Loaded .pt file: {len(data_truth)} events.")
    
    # 初始化总效率
    total_efficiency_edge = 0
    total_efficiency_track = 0  
    total_just_track = 0
    valid_events = 0
    valid_events_fit_track = 0  
    valid_events_just_track = 0

    # 遍历每个事件，计算效率
    for event_id in range(min(num_events, len(analyzed_tracks_list), len(data_truth))):
        event_tracks = analyzed_tracks_list[event_id]
        truth_data = data_truth[event_id]

        # 获取真值边的数量
        truth_edge = 0
        truth_track = 0
        good_fit = 0
        good_fit_track = 0
        just_track = 0

        # 遍历真值边
        truth_edges = []
        truth_track = count_tracks_in_truth(truth_data)  # 计算满足条件的轨迹数量
        for edge_idx, edge in enumerate(truth_data.edge_index.T):  # 转置以获取每条边
            if truth_data.y[edge_idx].item() == 1:  # 只考虑 y=1 的边
                truth_edge += 1

                # 获取真值边的两个节点
                node1_truth = truth_data.x[edge[0]]
                node2_truth = truth_data.x[edge[1]]
                truth_edges.append((node1_truth, node2_truth))  # 保存符合条件的 truth 边


                # 遍历 .lt 文件中的轨迹，寻找匹配的边
                for track in event_tracks:
                    if not hasattr(track, 'all_hits') or len(track.all_hits) < 2 or not track.above_four_hits:
                    # if not hasattr(track, 'all_hits') or len(track.all_hits) < 2:
                        continue  # 跳过没有足够 hits 的轨迹

                    # 遍历轨迹中的所有边
                    for i in range(len(track.all_hits) - 1):
                        node1_recon = track.all_hits[i]
                        node2_recon = track.all_hits[i + 1]
                        # 判断两个节点是否在范围内
                        if (
                            abs(node1_truth[0].item() - node1_recon[0]) <= range_threshold and
                            abs(node1_truth[2].item() - node1_recon[2]) <= range_threshold and
                            abs(node2_truth[0].item() - node2_recon[0]) <= range_threshold and
                            abs(node2_truth[2].item() - node2_recon[2]) <= range_threshold
                        ):
                            good_fit += 1
                            break  # 找到匹配的边后跳出循环

        # 遍历 .lt 文件中的轨迹，寻找匹配的边
        for track in event_tracks:
            if not hasattr(track, 'all_hits') or len(track.all_hits) < 2 or not track.above_four_hits:
                continue  # 跳过没有足够 hits 的轨迹
            just_track += 1
            track_good_fit = True  # 假设当前轨迹是 good_fit_track
            for i in range(len(track.all_hits) - 1):
                node1_recon = track.all_hits[i]
                node2_recon = track.all_hits[i + 1]

                # 判断当前 recon 边是否与任意 truth 边匹配
                edge_good_fit = False
                for node1_truth, node2_truth in truth_edges:
                    if (
                        abs(node1_truth[0].item() - node1_recon[0]) <= range_threshold and
                        abs(node1_truth[2].item() - node1_recon[2]) <= range_threshold and
                        abs(node2_truth[0].item() - node2_recon[0]) <= range_threshold and
                        abs(node2_truth[2].item() - node2_recon[2]) <= range_threshold
                    ):
                        edge_good_fit = True  # 当前 recon 边匹配成功
                        break

                if not edge_good_fit:
                    track_good_fit = False  # 如果有一条边不匹配，则整个轨迹不符合条件
                    break

            if track_good_fit:
                good_fit_track += 1  # 如果轨迹上的所有边都满足条件，则计为 good_fit_track

        # if event_id < 20:
        #     print(f"Event {event_id}: truth_track = {truth_track}, good_fit_track = {good_fit_track}, finding_track = {just_track}")

        if truth_track > 0:
            event_efficiency_track = good_fit_track / truth_track
            if event_efficiency_track > 1:
                # print(f"Event {event_id}: good_fit_track = {good_fit_track}, truth_track = {truth_track}, efficiency = {event_efficiency_track:.2%}")
                total_efficiency_track += 0
                valid_events_fit_track += 0
            else: 
                total_efficiency_track += event_efficiency_track 
                valid_events_fit_track += 1
        else:
            continue

        if truth_edge > 0:
            event_efficiency_edge = good_fit / truth_edge
            if event_efficiency_edge > 1:
                total_efficiency_edge += 0
                valid_events += 0
            else:
                total_efficiency_edge += event_efficiency_edge
                valid_events += 1
        else:
            continue

        if truth_track > 0:
            event_just_track = just_track / truth_track
            if event_just_track > 1:
                # print(f"Event {event_id}: just_track = {just_track}, truth_track = {truth_track}, efficiency = {event_just_track:.2%}")
                valid_events_just_track += 0
                total_just_track += 0
            else:
                total_just_track += event_just_track
                valid_events_just_track += 1
        else:
            continue


    # 计算平均效率
    average_efficiency_edge = total_efficiency_edge / valid_events if valid_events > 0 else 0
    average_efficiency_fit_track = total_efficiency_track / valid_events_fit_track if valid_events_fit_track > 0 else 0
    average_efficiency_find_track = total_just_track / valid_events_just_track if valid_events_just_track > 0 else 0
    print(f"Average Efficiency over {valid_events} events: edge: {average_efficiency_edge:.2%}")
    print(f"Average Efficiency over {valid_events_fit_track} events: fitting track: {average_efficiency_fit_track:.2%}")
    print(f"Average Efficiency over {valid_events_just_track} events: finding track: {average_efficiency_find_track:.2%}")


if __name__ == '__main__':
    file_path_recon = '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.momentum.rec.0p999/DigitizedRecTrk/tracks_0.lt'  
    file_path_raw = '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.momentum.rec.0p999/DigitizedRecTrk/momentum_0.pt'
    file_path_truth = '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.link.rec/DigitizedRecTrk/graph_1.pt'
    file_path_template = '/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/apply.link.rec/DigitizedRecTrk/graph_{}.pt'
    # for i in range(46):
    #     file_path_truth = file_path_template.format(i)
    #     if os.path.exists(file_path_truth):
    #         print(f"graph_id: {i}")
    #         efficiency_calculation(file_path_recon, file_path_truth, range_threshold=0.001, num_events=25)
    #     else:
    #         print(f"File {file_path_truth} does not exist.")
    # load_and_plot_recon_tracks_by_event(file_path_recon, file_path_truth, raw_hits=True, num_events=15)
    # load_and_plot_yz(file_path_recon, file_path_truth, raw_hits=True, num_events=15)
    # threeD_plot(file_path_recon, file_path_truth, raw_hits=True, num_events=15)
    efficiency_calculation(file_path_recon, file_path_truth, range_threshold=0.001, num_events=100)
    # inspect_file_structure(file_path_recon)
    # inspect_file_structure(file_path_truth)
    # print_position(file_path_recon, file_path_truth, raw_hits=True, num_events=5)

