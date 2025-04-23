import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import export
from typing import List, Dict, Tuple, Any, Set, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_linear_layers_in_graph(model: nn.Module, args, kwargs={}):
    """
    输入：
        model, args, kwargs, 模型前向可以使用这个进行： model(*args, **kwargs)
    
    作用：
        找出图中所有的线性层及其前后节点
    
    输出：
        linear_nodes: list[dict], 线性层节点信息，包含 linear 层前后的节点
            每个元素是一个字典：{
                'linear_node': node,
                'prev_nodes': List[node], 
                'after_nodes': List[node]
            }
    """
    try:
        model(*args, **kwargs)
    except Exception as e:
        logger.error(f"Model execution failed: {e}")
        raise RuntimeError("Model execution failed. Please check the model and input.")
    
    # Step 1: Determine which approach to use based on model complexity
    try:
        # Try using FX graph mode
        return _find_linear_layers_with_fx(model, args, kwargs)
    except Exception as e:
        logger.info(f"FX graph mode failed: {e}. Trying TorchDynamo approach.")
        try:
            # Try using TorchDynamo if FX fails
            return _find_linear_layers_with_dynamo(model, args, kwargs)
        except Exception as e:
            logger.error(f"Both FX and TorchDynamo approaches failed: {e}")
            raise RuntimeError("Failed to trace model with both FX and TorchDynamo.")

def _find_linear_layers_with_fx(model: nn.Module, args, kwargs={}):
    """Use torch.fx to find linear layers and their connections."""
    # Create a symbolic trace of the model
    traced_model = fx.symbolic_trace(model)
    graph = traced_model.graph
    
    # Dictionary to store nodes by name for easy lookup
    nodes_dict = {node.name: node for node in graph.nodes}
    
    # Dictionary to track input and output connections
    connections = {node.name: {'inputs': [], 'outputs': []} for node in graph.nodes}
    
    # Build the connection map
    for node in graph.nodes:
        for input_node in node.args:
            if hasattr(input_node, 'name') and input_node.name in connections:
                connections[node.name]['inputs'].append(input_node.name)
                connections[input_node.name]['outputs'].append(node.name)
    
    # Find linear layers
    linear_nodes = []
    for node in graph.nodes:
        if node.op == 'call_module':
            target_module = traced_model.get_submodule(node.target)
            if isinstance(target_module, nn.Linear):
                # Get previous nodes (inputs to this linear layer)
                prev_nodes = [nodes_dict[name] for name in connections[node.name]['inputs']]
                
                # Get next nodes (where this linear layer outputs to)
                after_nodes = [nodes_dict[name] for name in connections[node.name]['outputs']]
                
                linear_nodes.append({
                    'linear_node': node,
                    'prev_nodes': prev_nodes,
                    'after_nodes': after_nodes,
                    'module': target_module
                })
    
    return linear_nodes

def _find_linear_layers_with_dynamo(model: nn.Module, args, kwargs={}):
    """Use TorchDynamo to find linear layers and their connections."""
    # Export the model with TorchDynamo - using updated API
    exported_program = export(model)(*args, **kwargs)
    fx_graph = exported_program.graph_module
    graph = fx_graph.graph
    
    # Dictionary to store nodes by name for easy lookup
    nodes_dict = {node.name: node for node in graph.nodes}
    
    # Analyze the graph to find connections between nodes
    connections = {node.name: {'inputs': [], 'outputs': []} for node in graph.nodes}
    
    # Build connection map
    for node in graph.nodes:
        for arg in node.all_input_nodes:
            connections[node.name]['inputs'].append(arg.name)
            connections[arg.name]['outputs'].append(node.name)
    
    # Find linear layers
    linear_nodes = []
    for node in graph.nodes:
        # Check for linear operations
        if node.op == 'call_module':
            target_module = fx_graph.get_submodule(node.target)
            if isinstance(target_module, nn.Linear):
                # Get previous nodes
                prev_nodes = [nodes_dict[name] for name in connections[node.name]['inputs']]
                
                # Get next nodes
                after_nodes = [nodes_dict[name] for name in connections[node.name]['outputs']]
                
                linear_nodes.append({
                    'linear_node': node,
                    'prev_nodes': prev_nodes,
                    'after_nodes': after_nodes,
                    'module': target_module
                })
        elif node.op == 'call_function' and 'linear' in str(node.target):
            # Handle functional linear layers
            prev_nodes = [nodes_dict[name] for name in connections[node.name]['inputs']]
            after_nodes = [nodes_dict[name] for name in connections[node.name]['outputs']]
            
            linear_nodes.append({
                'linear_node': node,
                'prev_nodes': prev_nodes,
                'after_nodes': after_nodes,
                'module': None  # Functional, no module
            })
    
    return linear_nodes

def print_linear_layer_info(linear_nodes):
    """Helper function to print information about discovered linear layers."""
    print(f"Found {len(linear_nodes)} linear layers:")
    
    for idx, node_info in enumerate(linear_nodes):
        node = node_info['linear_node']
        module = node_info.get('module')
        
        # Print linear layer info
        if module:
            print(f"\n{idx+1}. Linear Layer: {node.name} - {module}")
            print(f"   Shape: in_features={module.in_features}, out_features={module.out_features}")
        else:
            print(f"\n{idx+1}. Functional Linear Layer: {node.name} - {node.target}")
        
        # Print previous nodes
        print(f"   Input nodes ({len(node_info['prev_nodes'])}):")
        for i, prev in enumerate(node_info['prev_nodes']):
            print(f"      {i+1}. {prev.name} (op: {prev.op})")
        
        # Print next nodes
        print(f"   Output nodes ({len(node_info['after_nodes'])}):")
        for i, after in enumerate(node_info['after_nodes']):
            print(f"      {i+1}. {after.name} (op: {after.op})")


# Example usage
if __name__ == "__main__":
    # Simple model for testing
    from tests.model import MiniLLM, TestModel
    
    model = TestModel()
    args = (torch.randn(1, 10), )
    
    model = MiniLLM()
    dummy_input = torch.randint(0, 100, (1, 10))  # batch_size=1, seq_len=10
    attention_mask = torch.ones(1, 10)  # Create appropriate attention mask
    args = (dummy_input, attention_mask)

    linear_nodes = find_linear_layers_in_graph(model, args)
    print_linear_layer_info(linear_nodes)