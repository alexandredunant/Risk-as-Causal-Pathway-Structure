import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# ==========================================
# 1. THE RULE REGISTRY (DNA)
# ==========================================
# These rules define how a State transforms into Future States.

def rule_stress_entropy(state_edges):
    """
    Universal Rule: Entropy deletes connections.
    Returns: List of possible next states (futures).
    """
    futures = []
    # Branching: The system could lose ANY one of its current edges
    if not state_edges:
        return []
        
    for edge in state_edges:
        new_edges = set(state_edges)
        new_edges.remove(edge)
        futures.append(frozenset(new_edges))
    return futures

def rule_repair_adaptation(state_edges):
    """
    Plant Rule: The system uses existing connections to build new ones (Transitivity).
    If A-B and B-C exist, grow A-C.
    """
    futures = []
    # Reconstruct the graph to find neighbors
    G = nx.Graph(list(state_edges))
    nodes = list(G.nodes())
    
    # Try to close triangles (local repair)
    for n in nodes:
        neighbors = list(G.neighbors(n))
        for n1, n2 in combinations(neighbors, 2):
            if not G.has_edge(n1, n2):
                new_edges = set(state_edges)
                new_edges.add((n1, n2))
                futures.append(frozenset(new_edges))
    return futures

SPECIES_DNA = {
    "glass": {
        "rules": [rule_stress_entropy], # Glass only degrades
        "color": "salmon" # Using "Risk" colors
    },
    "plant": {
        "rules": [rule_stress_entropy, rule_repair_adaptation], # Plant degrades AND repairs
        "color": "teal"
    }
}

# ==========================================
# 2. THE MULTIWAY EVOLVER
# ==========================================

def generate_multiway_graph(species_type, steps=20):
    """
    Generates the Map of Futures (Multiway Graph).
    Nodes = States (Configurations).
    Edges = Transitions (Time).
    """
    # 1. The Unbiased Seed: A simple Ring of 3 nodes
    # This is the "Present Moment" for both species.
    initial_state = frozenset({(1,2), (2,3), (3,1)})
    
    # The Multiway Graph tracks STATES, not atoms.
    MWG = nx.DiGraph()
    MWG.add_node(initial_state, layer=0, label="Start")
    
    frontier = [initial_state]
    seen_states = {initial_state}
    
    for step in range(1, steps + 1):
        new_frontier = []
        
        for current_state in frontier:
            # Check if dead (Empty set)
            if not current_state:
                continue

            # Get DNA
            rules = SPECIES_DNA[species_type]["rules"]
            
            # Apply all rules to generate all possible futures
            possible_futures = []
            for rule in rules:
                possible_futures.extend(rule(current_state))
            
            # Add to Multiway Graph
            for future_state in possible_futures:
                if future_state not in seen_states:
                    MWG.add_node(future_state, layer=step)
                    seen_states.add(future_state)
                    new_frontier.append(future_state)
                
                MWG.add_edge(current_state, future_state)
        
        frontier = new_frontier
        
    return MWG

# ==========================================
# 3. VISUALIZATION
# ==========================================

def get_layout(G):
    """
    Helper to safely select the best available layout.
    Tries Graphviz (Dot) first for the 'Tree' look, falls back to Spring.
    """
    try:
        # Try using pygraphviz or pydot
        return nx.nx_agraph.graphviz_layout(G, prog="dot")
    except (ImportError, AttributeError):
        try:
            # Fallback for older networkx versions or pydot interface
            return nx.nx_pydot.graphviz_layout(G, prog="dot")
        except:
            # Final fallback if no graphviz tools are installed
            return nx.spring_layout(G, seed=42)

def plot_futures(steps):
    # Generate the graphs
    # Note: We keep steps small because Multiway graphs grow exponentially
    print("Generating Glass Futures...")
    mwg_glass = generate_multiway_graph("glass", steps=steps)
    print("Generating Plant Futures...")
    mwg_plant = generate_multiway_graph("plant", steps=steps)

    plt.figure(figsize=(14, 7))

    # --- PLOT GLASS FUTURES ---
    plt.subplot(1, 2, 1)
    pos_glass = get_layout(mwg_glass)
    
    nx.draw(mwg_glass, pos_glass, node_size=50, node_color=SPECIES_DNA["glass"]["color"], 
            alpha=0.7, edge_color="gray", arrowsize=15)
    plt.title(f"Glass Multiway Graph (The Funnel)\nNodes: {mwg_glass.number_of_nodes()} | Edges: {mwg_glass.number_of_edges()}\nStructure: Convergent / Finite")

    # --- PLOT PLANT FUTURES ---
    plt.subplot(1, 2, 2)
    pos_plant = get_layout(mwg_plant)
        
    nx.draw(mwg_plant, pos_plant, node_size=50, node_color=SPECIES_DNA["plant"]["color"], 
            alpha=0.7, edge_color="gray", arrowsize=15)
    plt.title(f"Plant Multiway Graph (The Web)\nNodes: {mwg_plant.number_of_nodes()} | Edges: {mwg_plant.number_of_edges()}\nStructure: Divergent / Cyclic / Degenerate")

    plt.tight_layout()
    plt.savefig("multiway_futures_comparison.png")
    print("Plot saved to multiway_futures_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_futures(100)
