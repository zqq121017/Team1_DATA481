import os

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: 'graphviz' is not installed. Please install it using 'pip install graphviz'.")

def draw_model():
    if not HAS_GRAPHVIZ:
        return
    
    dot = Digraph(comment='GCN Model Architecture', format='png')
    dot.attr(rankdir='TB', size='10,10', bgcolor='white')

    # Define Node Styles
    dot.attr('node', style='filled', fontname='Arial', shape='box', rounded='true')
    
    # Inputs
    dot.node('EEG_In', 'EEG Data\\n(3 channels x 2 bands x Time x Freq)', fillcolor='lightblue')
    dot.node('Act_In', 'Action Sequence\\n(1 x 90 timesteps)', fillcolor='lightgreen')
    dot.node('Pup_In', 'Pupil Sequence\\n(1 x 90 timesteps)', fillcolor='lightpink')

    # Encoders
    dot.node('EEG_CNN', '2D-CNN Encoder\\n(Conv2d -> Pool -> Linear)', fillcolor='lightgrey')
    dot.node('Act_CNN', '1D-CNN Encoder\\n(Conv1d -> Pool -> Linear)', fillcolor='lightgrey')
    dot.node('Pup_CNN', '1D-CNN Encoder\\n(Conv1d -> Pool -> Linear)', fillcolor='lightgrey')

    # Extracted Nodes
    dot.node('EEG_Nodes', '3 Brain Nodes\\n(Fz, Cz, POz) [Embed Dim: 64]', fillcolor='#ffffcc')
    dot.node('Act_Node', '1 Behavior Node\\n[Embed Dim: 64]', fillcolor='#ffffcc')
    dot.node('Pup_Node', '1 Autonomic Node\\n[Embed Dim: 64]', fillcolor='#ffffcc')

    # Graph Formation
    dot.node('Graph_Concat', 'Concatenate 5 Nodes\\n(5 x 64 matrix)', fillcolor='#ffcc99', shape='ellipse')
    dot.node('Adj_Matrix', 'Adaptive Adjacency Matrix\\n(5x5 Learned Parameters)', fillcolor='#ff9999', shape='note')

    # Graph Convolution
    dot.node('GCN_1', 'Graph Convolution Layer 1\\n(Message Passing + ReLU)', fillcolor='#c2c2f0')
    dot.node('GCN_2', 'Graph Convolution Layer 2\\n(Message Passing + ReLU)', fillcolor='#c2c2f0')

    # Pooling and FC
    dot.node('Mean_Pool', 'Graph-Level Mean Pooling\\n(Average across all 5 nodes)', fillcolor='#ffb3e6', shape='ellipse')
    dot.node('FC_Head', 'Fully Connected Head\\n(Linear -> Dropout -> Linear)', fillcolor='#ffb3e6')
    dot.node('Output', 'Predicted Trial Score', fillcolor='#ff6666', shape='oval')

    # Edges
    dot.edge('EEG_In', 'EEG_CNN')
    dot.edge('Act_In', 'Act_CNN')
    dot.edge('Pup_In', 'Pup_CNN')

    dot.edge('EEG_CNN', 'EEG_Nodes')
    dot.edge('Act_CNN', 'Act_Node')
    dot.edge('Pup_CNN', 'Pup_Node')

    dot.edge('EEG_Nodes', 'Graph_Concat')
    dot.edge('Act_Node', 'Graph_Concat')
    dot.edge('Pup_Node', 'Graph_Concat')

    dot.edge('Graph_Concat', 'GCN_1')
    dot.edge('Adj_Matrix', 'GCN_1')
    
    dot.edge('GCN_1', 'GCN_2')
    dot.edge('Adj_Matrix', 'GCN_2')

    dot.edge('GCN_2', 'Mean_Pool')
    dot.edge('Mean_Pool', 'FC_Head')
    dot.edge('FC_Head', 'Output')

    output_path = 'gcn_model_structure_clean'
    dot.render(output_path, cleanup=True)
    print(f"Clean model structure successfully drawn and saved to: {output_path}.png")

if __name__ == "__main__":
    draw_model()
