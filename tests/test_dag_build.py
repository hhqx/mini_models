import torch
import torch.nn as nn
import torch.fx
import os
import matplotlib.pyplot as plt
import networkx as nx
from torch.fx.graph_module import GraphModule
from model import create_mini_model, TransformerBlock, MiniLLM


def test_sequential_dag():
    """
    方法1：使用 PyTorch 的 nn.Sequential 构建 DAG
    这是一种更传统的方式，通过模块组合构建计算图
    """
    print("\n=== 方法1: 使用 nn.Sequential 构建 DAG ===")
    
    # 定义一个小型的序列模型，每一步都是明确的
    class SequentialTransformer(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64, num_heads=2, d_ff=128, num_layers=2):
            super().__init__()
            
            # 定义各个组件
            self.embedding = nn.Embedding(vocab_size, d_model)
            
            # 使用Sequential构建Transformer层
            layers = []
            for _ in range(num_layers):
                attention_block = nn.Sequential(
                    nn.LayerNorm(d_model),
                    MultiHeadAttentionWrapper(d_model, num_heads),
                    nn.Dropout(0.1)
                )
                
                ff_block = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(0.1)
                )
                
                layers.append(AttentionWithResidual(attention_block))
                layers.append(ResidualConnection(ff_block))
            
            self.transformer_layers = nn.Sequential(*layers)
            self.output_projection = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer_layers(x)
            return self.output_projection(x)
    
    # 辅助类用于实现残差连接
    class ResidualConnection(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, x):
            return x + self.module(x)
    
    # 特殊的包装器来处理自注意力
    class MultiHeadAttentionWrapper(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            
        def forward(self, x):
            output, _ = self.attn(x, x, x)
            return output
    
    # 包装自注意力并添加残差连接
    class AttentionWithResidual(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, x):
            return x + self.module(x)
    
    # 创建模型实例
    model = SequentialTransformer()
    
    # 测试模型
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # 前向传播
    output = model(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    return model


def test_fx_dag():
    """
    方法2: 使用 PyTorch FX 进行计算图跟踪和转换
    这种方法允许对计算图进行更高级的分析和优化
    """
    print("\n=== 方法2: 使用 PyTorch FX 构建 DAG ===")
    
    # 创建一个简单的模型
    model = create_mini_model(vocab_size=1000, max_seq_len=64)
    
    # 准备输入示例用于跟踪
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    
    # 使用FX进行符号跟踪
    try:
        # 创建一个符号跟踪的模型版本
        traced_model = torch.fx.symbolic_trace(model)
        
        # 显示图结构
        print("\nFX计算图模块:")
        print(traced_model.graph)
        
        # 打印节点
        print("\nFX图节点:")
        for node in traced_model.graph.nodes:
            print(f"节点: {node.name}, 操作: {node.op}, 目标: {node.target}")
        
        # 运行追踪后的图
        output_traced = traced_model(input_ids, attention_mask)
        print(f"\nFX追踪后的输出形状: {output_traced.shape}")
        
        return traced_model
        
    except Exception as e:
        print(f"FX追踪失败，可能是因为模型包含动态控制流: {str(e)}")
        
        # 如果完整模型不能跟踪，尝试跟踪单个Transformer块
        print("\n尝试跟踪单个Transformer块...")
        block = TransformerBlock(d_model=128, num_heads=4, d_ff=256)
        x = torch.randn(batch_size, seq_len, 128)
        
        try:
            traced_block = torch.fx.symbolic_trace(block)
            print("成功追踪单个Transformer块")
            print(traced_block.graph)
            return traced_block
        except Exception as e2:
            print(f"单个块追踪也失败: {str(e2)}")
            return None


def visualize_model_graph(graph_module, filename="model_graph"):
    """
    使用 NetworkX 和 Matplotlib 可视化模型的计算图
    """
    G = nx.DiGraph()
    
    # 添加节点
    for node in graph_module.graph.nodes:
        G.add_node(node.name, label=f"{node.name}\n{node.op}\n{node.target.__name__ if callable(node.target) else str(node.target)}")
        
        # 添加边
        for input_node in node.all_input_nodes:
            G.add_edge(input_node.name, node.name)
    
    # 绘制图形
    plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(G, seed=42)  # 使用spring布局算法
    
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=node_labels, 
            node_color="lightblue", node_size=2000, font_size=8,
            arrows=True, arrowsize=20, edge_color="gray")
    
    plt.title("Model Computation Graph")
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {filename}.png")
    plt.close()
    
    return G


def trace_and_visualize_model():
    """
    创建模型，追踪并可视化计算图
    """
    # 创建简化版模型用于追踪
    class SimplifiedTransformer(nn.Module):
        def __init__(self, d_model=64, d_ff=128):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(d_ff, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
            
        def forward(self, x):
            residual = x
            x = self.linear1(x)
            x = self.gelu(x)
            x = self.linear2(x)
            x = x + residual
            x = self.layer_norm(x)
            return x
    
    model = SimplifiedTransformer()
    
    # 创建样例输入
    x = torch.randn(2, 10, 64)
    
    # 追踪模型
    traced_model = torch.fx.symbolic_trace(model)
    
    # 可视化
    graph = visualize_model_graph(traced_model, "simplified_transformer_graph")
    
    # 返回追踪后的模型和图
    return traced_model, graph


def find_linear_layers_in_graph(graph_module):
    """
    找出图中所有的线性层及其前后节点
    """
    linear_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            target_module = graph_module.get_submodule(node.target)
            if isinstance(target_module, nn.Linear):
                # 找到前置节点
                prev_nodes = list(node.all_input_nodes)
                # 找到后续节点
                next_nodes = []
                for n in graph_module.graph.nodes:
                    if node in n.all_input_nodes:
                        next_nodes.append(n)
                
                linear_nodes.append({
                    'node': node,
                    'module': target_module,
                    'prev_nodes': prev_nodes,
                    'next_nodes': next_nodes
                })
                
                print(f"找到线性层: {node.target}")
                print(f"  输入节点: {[n.name for n in prev_nodes]}")
                print(f"  输出节点: {[n.name for n in next_nodes]}")
    
    return linear_nodes


if __name__ == "__main__":
    print("测试小型LLM模型和DAG构建方法...")
    
    # 测试常规模型
    # mini_model = create_mini_model()
    # batch_size, seq_len = 2, 10
    # input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    # attention_mask = torch.ones((batch_size, seq_len))
    
    # output = mini_model(input_ids, attention_mask)
    # print(f"\n基本模型测试:")
    # print(f"输入形状: {input_ids.shape}")
    # print(f"输出形状: {output.shape}")
    # print(f"模型参数总数: {sum(p.numel() for p in mini_model.parameters())}")
    
    # # 测试两种DAG构建方法
    # sequential_model = test_sequential_dag()
    # fx_model = test_fx_dag()
    
    # 追踪并可视化模型
    print("\n=== 可视化模型计算图 ===")
    traced_model, graph = trace_and_visualize_model()
    
    # 找出所有线性层
    print("\n=== 查找线性层及其连接关系 ===")
    linear_layers = find_linear_layers_in_graph(traced_model)
    
    print("\n完成测试!")
