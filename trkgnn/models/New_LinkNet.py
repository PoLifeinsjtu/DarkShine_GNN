import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add

from .utils import make_mlp
from utility.FunctionTime import timing_decorator

class CBAM(nn.Module):
    def __init__(self, dim, reduction_ratio=16):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        # 通道注意力分支
        self.channel_attn = nn.Sequential(
            nn.Linear(2 * dim, dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(dim // reduction_ratio, dim),
            nn.Sigmoid()
        )

        # 空间注意力分支（适用于序列数据）
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),  
            nn.Sigmoid()
        )

    @timing_decorator
    def forward(self, x):
        # 检查输入范围是否合理
        # print("Input min/max:", x.min().item(), x.max().item())
        # x shape: (N, C) 或 (C,) → 添加 batch 维度
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 通道注意力
        max_pool = torch.max(x, dim=0, keepdim=True)[0]  # (1, C)
        avg_pool = torch.mean(x, dim=0, keepdim=True)    # (1, C)
        cat_pool = torch.cat([max_pool, avg_pool], dim=1)  # (1, 2*C)
        channel_weights = self.channel_attn(cat_pool)      # (1, C)
        x = x * channel_weights.expand_as(x)               # (N, C)

        # 空间注意力（适用于序列数据）
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, C)
        max_pool = torch.max(x, dim=-1, keepdim=True)[0]  # (1, N, 1)
        avg_pool = torch.mean(x, dim=-1, keepdim=True)     # (1, N, 1)
        cat_pool = torch.cat([max_pool, avg_pool], dim=-1)  # (1, N, 2)

        cat_pool = cat_pool.permute(0, 2, 1)  # (1, 2, N)
        spatial_weights = self.spatial_attn(cat_pool)  # (1, 1, N)
        x = x * spatial_weights.permute(0, 2, 1).expand_as(x)  # (1, N, C)

        return x.squeeze(0)  # 去掉 batch 维度（如果有的话）

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    @timing_decorator
    def forward(self, x1, x2):
        gate_weights = self.gate(torch.cat([x1, x2], dim=1))
        return x1 * gate_weights + x2 * (1 - gate_weights)

class New_LinkNet(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_iterations=3):
        super(New_LinkNet, self).__init__()

        self.n_iterations = n_iterations

        # 使用 make_mlp 构建节点和边的嵌入层
        self.node_embedding = make_mlp(node_input_dim, [hidden_dim], dropout=0.1)
        self.edge_embedding = make_mlp(edge_input_dim + 3, [hidden_dim], dropout=0.1)  # +3 为 Δx, Δy, Δz

        self.z_values = torch.tensor([7.98, 22.98, 38.98, 53.98, 89.98, 180.48])
        # 嵌入层大小设为Z值数量（6）
        self.z_embedding = nn.Embedding(len(self.z_values), hidden_dim)
        
        # 预计算Z值到索引的映射（用于快速查找）
        self.z_to_index = {8: 0, 23: 1, 39: 2, 54: 3, 90: 4, 180: 5}

        # 通道注意力模块
        self.cbam_node = CBAM(hidden_dim)
        self.cbam_edge = CBAM(hidden_dim)

        # Transformer 层
        self.transformer_conv_list = nn.ModuleList([
            TransformerConv(2 * hidden_dim, hidden_dim, heads=heads) for _ in range(n_iterations)
        ])

        self.norm_transconv_list = nn.ModuleList([
            nn.LayerNorm(heads * hidden_dim) for _ in range(n_iterations)
        ])
        self.norm_combined_list = nn.ModuleList([
            nn.LayerNorm(2 * hidden_dim) for _ in range(n_iterations)
        ])

        self.projection_layer_edge_list = nn.ModuleList([
            nn.Linear(heads * hidden_dim, hidden_dim) for _ in range(n_iterations)
        ])
        self.projection_layer_node_list = nn.ModuleList([
            nn.Linear(heads * hidden_dim, hidden_dim) for _ in range(n_iterations)
        ])

        self.gfm_list = nn.ModuleList([
            GatedFusion(hidden_dim) for _ in range(n_iterations)
        ])

        # 边分类器
        self.edge_classifier = make_mlp(2 * hidden_dim, [hidden_dim, 1], output_activation=None)

    @timing_decorator
    def forward(self, data):
        send_idx = torch.cat([data.edge_index[0], data.edge_index[1]], dim=0)
        recv_idx = torch.cat([data.edge_index[1], data.edge_index[0]], dim=0)
        edge_indices = torch.stack([send_idx, recv_idx], dim=0)
        edge_attr_bi = torch.cat([data.edge_attr, data.edge_attr], dim=0)        
        src_indices, dst_indices = edge_indices

        src_pos = data.x[src_indices][:, :3]
        dst_pos = data.x[dst_indices][:, :3]
        rel_pos = src_pos - dst_pos

        # 从data.x中提取Z坐标值（第3列，索引为2）
        z_values = data.x[:, 2]  
        z_integers = torch.round(z_values).long()
        valid_mask = torch.tensor([z.item() in self.z_to_index for z in z_integers], device=z_values.device)
        
        if not valid_mask.all():
            invalid_indices = torch.where(~valid_mask)[0]
            invalid_z_values = z_integers[invalid_indices]
            raise ValueError(f"Invalid Z values found: {invalid_z_values.tolist()}. Allowed integers: {list(self.z_to_index.keys())}")
        
        # 转换Z值为对应的索引
        z_indices = torch.tensor([self.z_to_index[z.item()] for z in z_integers], device=z_values.device)

        node_features = self.node_embedding(data.x) + self.z_embedding(z_indices)
        node_features = self.cbam_node(node_features)

        edge_features = torch.cat([edge_attr_bi, rel_pos], dim=1)
        edge_features = self.edge_embedding(edge_features)
        edge_features = self.cbam_edge(edge_features)

        for i, (transformer_conv, norm_combined, norm_transconv,
                projection_layer_edge, projection_layer_node, gfm) in enumerate(zip(
            self.transformer_conv_list,
            self.norm_combined_list,
            self.norm_transconv_list,
            self.projection_layer_edge_list,
            self.projection_layer_node_list,
            self.gfm_list
        )):
            x0 = node_features
            e0 = edge_features

            aggregated_from_src = scatter_add(edge_features, dst_indices, dim=0, dim_size=node_features.shape[0])
            combined_features = torch.cat([node_features, aggregated_from_src - node_features], dim=1)
            combined_features = norm_combined(combined_features)

            out_node_features = transformer_conv(combined_features, edge_indices)
            out_node_features = norm_transconv(out_node_features)

            node_features = gfm(node_features, projection_layer_node(out_node_features))
            edge_features = projection_layer_edge(out_node_features[src_indices] - out_node_features[dst_indices])

            node_features = node_features + x0
            edge_features = edge_features + e0

        start_idx, end_idx = data.edge_index
        clf_inputs = torch.cat([node_features[start_idx], node_features[end_idx]], dim=1)
        edge_output = self.edge_classifier(clf_inputs).squeeze(-1)

        return edge_output

class TrkTrans(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_iterations=3):
        super(TrkTrans, self).__init__()
        self.link = New_LinkNet(node_input_dim, edge_input_dim, hidden_dim, heads, n_iterations)

    @timing_decorator
    def forward(self, data):
        edge_scores = self.link(data)
        return edge_scores

def build_model(**kwargs):
    return TrkTrans(**kwargs)