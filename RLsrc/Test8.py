import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.DiGraph()

# Add nodes and edges with coordinates
G.add_node("A", pos=(0, 0))
G.add_node(2, pos=(1, 0))
G.add_node(3, pos=(1, 1))
G.add_node(4, pos=(2, 0))
G.add_node(5, pos=(2, 1))
G.add_edges_from([("A", 2), ("A", 3), (2, 4), (2, 5)])

# Get the positions of nodes
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_weight='bold', width=2,
        arrowstyle='<-', arrowsize=15)
plt.axis('equal')
plt.show()
