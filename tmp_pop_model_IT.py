import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from collections import defaultdict

# ==========================================
# 1. TOPOLOGICAL RULES (Deterministic)
# ==========================================

def get_successors(state):
    """
    Returns all topologically possible next states based on discrete rules.
    State is a frozenset of edges.
    """
    successors = []
    
    # --- Rule 1: Fertility Decline (Entropy) ---
    # Any existing connection can fail.
    if len(state) > 0:
        for edge in state:
            next_state = set(state)
            next_state.remove(edge)
            successors.append(frozenset(next_state))
            
    # --- Rule 2: Technology Expansion (Repair/Growth) ---
    # Closes 'open triangles' (A-B, B-C -> add A-C).
    # We reconstruct the graph from edges to find neighbors
    G_temp = nx.Graph(list(state))
    nodes = list(G_temp.nodes())
    
    # Check all potential triangles
    if len(nodes) >= 3:
        for n1, n2 in combinations(nodes, 2):
            if not G_temp.has_edge(n1, n2):
                # If they share a neighbor (common)
                common = set(G_temp.neighbors(n1)) & set(G_temp.neighbors(n2))
                if common:
                    next_state = set(state)
                    next_state.add(tuple(sorted((n1, n2))))
                    successors.append(frozenset(next_state))
                
    # --- Rule 3: Systemic Shock (Collapse) ---
    # If system is highly connected (e.g., >4 edges), it risks cascading failure.
    if len(state) > 4:
        edges = list(state)
        # Deterministic: remove the first pair of connected edges found
        for i, e1 in enumerate(edges):
            for e2 in edges[i+1:]:
                if set(e1) & set(e2): # They share a node
                    next_state = set(state)
                    next_state.remove(e1)
                    next_state.remove(e2)
                    successors.append(frozenset(next_state))
                    break 
            else: continue
            break
    
    # Return unique outcomes only
    return list(set(successors))

# ==========================================
# 2. SIMULATION ENGINE (Corrected)
# ==========================================

def run_simulation(generations=12):
    # Initial State: Hexagon (6 edges) representing stability
    initial_state = frozenset({(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)})
    
    # 1. Structures for Graph Plotting
    MWG = nx.DiGraph()
    MWG.add_node(initial_state, layer=0, size=len(initial_state))
    
    # 2. Structures for Statistical Plotting (River/Prediction)
    # current_layer maps {state: path_count} to track topological multiplicity
    current_layer = {initial_state: 1} 
    
    history = [] 
    
    print(f"Simulating {generations} generations...")
    
    for t in range(generations):
        # --- Record Statistics for User's Plots ---
        size_dist = defaultdict(int)
        for state, path_count in current_layer.items():
            size = len(state) # Proxy for System Health
            size_dist[size] += path_count
            
        history.append({
            "time": t,
            "sizes": size_dist,
            "total_paths": sum(size_dist.values())
        })
        
        # --- Evolve to Next Layer ---
        next_layer = defaultdict(int)
        
        for state, path_count in current_layer.items():
            successors = get_successors(state)
            
            # If dead end, history stops here for this path (stasis)
            if not successors:
                next_layer[state] += path_count 
                continue
                
            for next_state in successors:
                # A. Update Multiway Graph (Structure)
                # FIX: Check existence and set attributes BEFORE adding edge
                if next_state not in MWG:
                    MWG.add_node(next_state, layer=t+1, size=len(next_state))
                
                MWG.add_edge(state, next_state)
                
                # B. Update Path Counts (Statistics)
                # Propagate the path multiplicity forward
                next_layer[next_state] += path_count
                
        current_layer = next_layer

    return history, MWG

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_results(history):
    """
    Plots the Topological River and Prediction Capacity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- PLOT 1: The Topological River ---
    max_time = len(history)
    # Determine max size dynamically for grid height
    all_sizes = set()
    for step in history:
        all_sizes.update(step["sizes"].keys())
    max_size = max(all_sizes) + 2 if all_sizes else 10
    
    grid = np.zeros((max_size, max_time))
    
    for t, data in enumerate(history):
        total = data["total_paths"]
        if total > 0:
            for size, count in data["sizes"].items():
                grid[size, t] = count / total

    sns.heatmap(grid, cmap="mako", ax=ax1, cbar_kws={'label': 'Pathway Density'})
    ax1.invert_yaxis()
    ax1.set_title("The Topological River: Convergence of Futures", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time (Generations)")
    ax1.set_ylabel("Population / Connectivity")
    
    # Trend line
    means = []
    for t in range(max_time):
        if history[t]["total_paths"] > 0:
            w_sum = sum(s * c for s, c in history[t]["sizes"].items())
            means.append(w_sum / history[t]["total_paths"])
        else:
            means.append(0)
            
    ax1.plot([x + 0.5 for x in range(max_time)], [y + 0.5 for y in means], 
             'w--', linewidth=2, label="Topological Center of Mass")
    ax1.legend()

    # --- PLOT 2: Capacity for Prediction ---
    times = [d["time"] for d in history]
    spreads = []
    for t in range(max_time):
        if history[t]["total_paths"] > 0:
            sizes = list(history[t]["sizes"].keys())
            counts = list(history[t]["sizes"].values())
            avg = means[t]
            variance = sum(c * ((s - avg) ** 2) for s, c in zip(sizes, counts)) / sum(counts)
            spreads.append(np.sqrt(variance))
        else:
            spreads.append(0)
    
    ax2.plot(times, spreads, 'o-', color='#e74c3c', linewidth=3)
    ax2.set_title("Inverse Capacity for Prediction (Uncertainty)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Spread of Possible Outcomes (Std Dev)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    
    # Annotate phases
    ax2.axvspan(0, 4, color='green', alpha=0.1, label='Deterministic Phase')
    ax2.axvspan(4, 9, color='yellow', alpha=0.1, label='Divergence Phase')
    ax2.axvspan(9, max_time-1, color='gray', alpha=0.1, label='Attractor Phase')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_clean_graph(MWG):
    """
    Clean, structural visualization of the Multiway Causal Graph.
    """
    plt.figure(figsize=(16, 10), facecolor='white')
    ax = plt.gca()
    
    # Layout Logic: X=Time, Y=Size (with jitter for clarity)
    pos = {}
    layer_counts = defaultdict(int)
    
    # We sort nodes to ensure consistent plotting order
    for node in sorted(MWG.nodes(), key=lambda n: (MWG.nodes[n].get('layer',0), MWG.nodes[n].get('size',0))):
        data = MWG.nodes[node]
        layer = data.get('layer', 0)
        size = data.get('size', 0)
        
        # Jitter logic to prevent overlap
        key = (layer, size)
        layer_counts[key] += 1
        y_base = size * 2
        # Alternating offset for nodes of same size at same time
        offset = (layer_counts[key] * 0.4) * (1 if layer_counts[key] % 2 == 0 else -1)
        
        pos[node] = (layer, y_base + offset)

    # Draw Edges (Curved)
    for u, v in MWG.edges():
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Color edge: Green (Growth) or Red (Decay) based on size change
            u_size = MWG.nodes[u]['size']
            v_size = MWG.nodes[v]['size']
            color = '#2ecc71' if v_size > u_size else '#e74c3c'
            
            ax.annotate("",
                        xy=(x2, y2), xycoords='data',
                        xytext=(x1, y1), textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", color=color, 
                                        alpha=0.3, shrinkA=5, shrinkB=5,
                                        connectionstyle="arc3,rad=0.1"))

    # Draw Nodes
    # Collect sizes for coloring
    sizes = [MWG.nodes[n]['size'] for n in MWG.nodes()]
    
    # Use a scatter plot for nodes to handle colormapping easily
    # We need to ensure the order matches 'pos' keys if iterating, 
    # but nx.draw_networkx_nodes handles this if we pass the nodelist.
    nx.draw_networkx_nodes(MWG, pos, 
                           node_size=120, 
                           node_color=sizes, 
                           cmap='magma', 
                           edgecolors='white', 
                           linewidths=1.5)

    plt.title("Structure of the Multiway Causal Graph", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Time (Generations)", fontsize=12)
    plt.ylabel("System Complexity (Connectivity)", fontsize=12)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Growth/Repair Pathway'),
        mpatches.Patch(color='#e74c3c', label='Decay/Shock Pathway'),
        mpatches.Circle((0,0), facecolor='purple', label='High Complexity State'),
        mpatches.Circle((0,0), facecolor='orange', label='Low Complexity State')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================

# 1. Run Simulation
history_data, multiway_graph = run_simulation(generations=12)

# 2. Plot User's Requested Analysis (River + Prediction)
plot_results(history_data)

# 3. Plot Clean Graph Structure
plot_clean_graph(multiway_graph)