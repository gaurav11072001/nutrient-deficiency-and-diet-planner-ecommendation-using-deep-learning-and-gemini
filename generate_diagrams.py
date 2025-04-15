import graphviz
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_context_dfd():
    dot = graphviz.Digraph('Context_DFD', comment='Level 0 DFD (Context Diagram)')
    dot.attr(rankdir='TB')
    
    # Add nodes
    dot.node('User', 'User', shape='circle')
    dot.node('System', 'Vitamin Deficiency\nPrediction System', shape='box')
    dot.node('DB', 'Database', shape='cylinder')
    
    # Add edges
    dot.edge('User', 'System', 'Input symptoms & info')
    dot.edge('System', 'User', 'Predictions & Diet Plans')
    dot.edge('System', 'DB', 'Store/Retrieve Data')
    dot.edge('DB', 'System', 'User Data & Results')
    
    # Save diagram
    dot.render('diagrams/context_dfd', format='png', cleanup=True)

def create_level1_dfd():
    dot = graphviz.Digraph('Level1_DFD', comment='Level 1 DFD')
    dot.attr(rankdir='TB')
    
    # Add nodes
    dot.node('User', 'User', shape='circle')
    dot.node('Auth', 'Authentication\nSystem', shape='box')
    dot.node('Predict', 'Prediction\nEngine', shape='box')
    dot.node('Diet', 'Diet\nRecommendation\nSystem', shape='box')
    dot.node('DB', 'Database', shape='cylinder')
    dot.node('Model', 'ML Model', shape='box')
    
    # Add edges
    dot.edge('User', 'Auth', 'Register/Login')
    dot.edge('Auth', 'DB', 'Verify')
    dot.edge('User', 'Predict', 'Input Symptoms')
    dot.edge('Predict', 'Model', 'Process Data')
    dot.edge('Model', 'Diet', 'Deficiency Prediction')
    dot.edge('Diet', 'User', 'Generate Plan')
    dot.edge('DB', 'DB', 'Store Results')
    
    # Save diagram
    dot.render('diagrams/level1_dfd', format='png', cleanup=True)

def create_er_diagram():
    dot = graphviz.Digraph('ER_Diagram', comment='Entity Relationship Diagram')
    dot.attr(rankdir='LR')
    
    # Add entity nodes
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Users')
        c.node('Users', '''Users
        ------------------
        + id (PK)
        + first_name
        + last_name
        + email
        + password
        + created_at''', shape='box')
    
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Predictions')
        c.node('Predictions', '''Predictions
        ------------------
        + id (PK)
        + user_id (FK)
        + deficiency_type
        + probability
        + prediction_date
        + diet_recommendation''', shape='box')
    
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Symptoms')
        c.node('Symptoms', '''Symptoms
        ------------------
        + id (PK)
        + prediction_id (FK)
        + symptom_name
        + is_present''', shape='box')
    
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Deficiency Types')
        c.node('DeficiencyTypes', '''Deficiency Types
        ------------------
        + id (PK)
        + name
        + description
        + common_symptoms''', shape='box')
    
    # Add relationships
    dot.edge('Users', 'Predictions', 'makes (1:n)')
    dot.edge('Predictions', 'Symptoms', 'has (1:n)')
    dot.edge('Predictions', 'DeficiencyTypes', 'predicts (n:1)')
    
    # Save diagram
    dot.render('diagrams/er_diagram', format='png', cleanup=True)

def create_system_architecture():
    dot = graphviz.Digraph('System_Architecture', comment='System Architecture')
    dot.attr(rankdir='TB')
    
    # Frontend subgraph
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Frontend')
        c.node('UI', 'User Interface')
        c.node('Forms', 'Input Forms')
        c.node('Charts', 'Visualization')
    
    # Backend subgraph
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Backend')
        c.node('Auth', 'Authentication')
        c.node('API', 'Flask API')
        c.node('ML', 'Machine Learning')
        c.node('Diet', 'Diet Generator')
    
    # Database subgraph
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Database')
        c.node('UserDB', 'User Data', shape='cylinder')
        c.node('PredDB', 'Predictions', shape='cylinder')
    
    # Add edges
    dot.edge('UI', 'Forms')
    dot.edge('Forms', 'API')
    dot.edge('API', 'Auth')
    dot.edge('Auth', 'UserDB')
    dot.edge('API', 'ML')
    dot.edge('ML', 'Diet')
    dot.edge('Diet', 'API')
    dot.edge('API', 'PredDB')
    dot.edge('API', 'Charts')
    
    # Save diagram
    dot.render('diagrams/system_architecture', format='png', cleanup=True)

def create_neural_network():
    G = nx.DiGraph()
    
    # Add nodes for each layer
    input_layer = ['I1', 'I2', 'I3']
    hidden_layer1 = ['H1_1', 'H1_2', 'H1_3']
    hidden_layer2 = ['H2_1', 'H2_2']
    output_layer = ['O1', 'O2']
    
    # Add nodes to graph
    G.add_nodes_from(input_layer)
    G.add_nodes_from(hidden_layer1)
    G.add_nodes_from(hidden_layer2)
    G.add_nodes_from(output_layer)
    
    # Add edges between layers
    for i in input_layer:
        for h in hidden_layer1:
            G.add_edge(i, h)
    
    for h1 in hidden_layer1:
        for h2 in hidden_layer2:
            G.add_edge(h1, h2)
    
    for h2 in hidden_layer2:
        for o in output_layer:
            G.add_edge(h2, o)
    
    # Create layout
    pos = {}
    layer_width = 2.0
    for i, node in enumerate(input_layer):
        pos[node] = (0, i - (len(input_layer)-1)/2.0)
    for i, node in enumerate(hidden_layer1):
        pos[node] = (layer_width, i - (len(hidden_layer1)-1)/2.0)
    for i, node in enumerate(hidden_layer2):
        pos[node] = (2*layer_width, i - (len(hidden_layer2)-1)/2.0)
    for i, node in enumerate(output_layer):
        pos[node] = (3*layer_width, i - (len(output_layer)-1)/2.0)
    
    # Draw the network
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, arrowsize=20, font_size=10,
            font_weight='bold')
    plt.title('Neural Network Architecture')
    plt.savefig('diagrams/neural_network.png')
    plt.close()

def main():
    # Create diagrams directory if it doesn't exist
    if not os.path.exists('diagrams'):
        os.makedirs('diagrams')
    
    # Generate all diagrams
    print("Generating Context DFD...")
    create_context_dfd()
    
    print("Generating Level 1 DFD...")
    create_level1_dfd()
    
    print("Generating ER Diagram...")
    create_er_diagram()
    
    print("Generating System Architecture...")
    create_system_architecture()
    
    print("Generating Neural Network Diagram...")
    create_neural_network()
    
    print("All diagrams have been generated in the 'diagrams' directory.")

if __name__ == "__main__":
    main() 