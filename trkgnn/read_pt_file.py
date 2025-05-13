import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import argparse

def plot_graph(file_path):
    # 加载模型
    data_list = torch.load(file_path, map_location='cpu', weights_only=False)
    
    # 检查并打印存在的属性或输出“无”
    # attributes_to_check = ['x', 'edge_index', 'y', 'w', 'n', 'i', 'run_num', 'evt_num', 'edge_attr', 'truth_w', 'p']
    # for attr in attributes_to_check:
    #     if hasattr(data_list[0], attr):
    #         tensor = getattr(data_list[0], attr)
    #         if attr == 'edge_attr':
    #             print(f'{attr}: exist, shape: {tensor.shape}, values: {tensor}')
    #         else:
    #             print(f'{attr}: exist, shape: {tensor.shape}')
    #     elif isinstance(data_list[0], dict) and attr in data_list[0]:
    #         tensor = data_list[0][attr]
    #         if attr == 'edge_attr':
    #             print(f'{attr}: exist, shape: {tensor.shape}, values: {tensor}')
    #         else:
    #             print(f'{attr}: exist, shape: {tensor.shape}')
    #     else:
    #         print(f'{attr}: None')

    for idx, data in enumerate(data_list[:10]):  # 只画前n张图
        # 转换并绘制图，按照坐标信息绘制
        G = to_networkx(data, to_undirected=True)  # 如果图是有向的，移除to_undirected参数

        # 获取节点的坐标信息
        if hasattr(data, 'x') and data.x is not None:
            pos = {i: (data.x[i][0].item(), data.x[i][2].item()) for i in range(data.num_nodes)}
        else:
            # 如果没有位置信息，则使用NetworkX的spring布局作为替代
            pos = nx.spring_layout(G, dim=2)

        # 绘制2D图形
        fig, ax = plt.subplots()
        for node, (x, z) in pos.items():
            ax.scatter(z, x, s=100, c='skyblue', edgecolors='k', alpha=0.7)
            ax.text(z, x, s=str(node), fontsize=10, fontweight='bold')

        for edge in G.edges():
            z = [pos[edge[0]][0], pos[edge[1]][0]]
            x = [pos[edge[0]][1], pos[edge[1]][1]]

            # 获取边的属性并计算透明度
            edge_index = list(G.edges()).index(edge)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                score = data.edge_attr[edge_index][3].item()  # 获取最后一个数作为score
                alpha = max(score, 0)  # 如果score是负数，则设置为0
                alpha = min(score, 1)
            else:
                alpha = 0  # 如果没有edge_attr，使用默认透明度

            ax.plot(x, z, c='black', alpha=alpha)
        
        # 添加轴标签
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("x (mm)")

        # 保存图形
        output_path = f"fig/truth/{file_path.split('/')[-1].replace('.pt', f'_{idx}.png')}"
        plt.savefig(output_path)
        plt.close()
        print(f"Graph {idx} saved to {output_path}")

    
def read_truth(file_path):
    data_list = torch.load(file_path)
    # 读取数据
    for i, data in enumerate(data_list):
        print(f"Graph {i}:")
        print(f"  Number of edges: {data.num_edges}")
        # print(f"  Truth labels (y): {data.y}")
        if hasattr(data, 'x') and data.x is not None:
            pos = {i: (data.x[i][0].item(), data.x[i][2].item()) for i in range(data.num_nodes)}
        else:
            # 如果没有位置信息，则使用NetworkX的spring布局作为替代
            pos = nx.spring_layout(G, dim=2)

        G = to_networkx(data, to_undirected=True)  # 如果图是有向的，移除to_undirected参数
        for edge in G.edges():
            edge_index = list(G.edges()).index(edge)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None and pos[edge[0]][1] < 8:
                score = data.edge_attr[edge_index][3].item()  # 获取最后一个数作为score
                if score > 0.5:
                    print(f"first edge pos x: {pos[edge[0]][0]}, score: {score}")

        if i > 1:
            break



if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Plot graph from a .pt file")
    parser.add_argument('file', type=str, help="the input .pt file")
    args = parser.parse_args()
    # plot_graph(args.file)
    read_truth(args.file)