import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import json
import numpy as np
import imageio.v2 as imageio
import tempfile
import shutil
import os

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
# 3. RISK METRICS & QUANTIFICATION
# ==========================================

def compute_risk_metrics(MWG, species_name):
    """
    Compute comprehensive risk metrics from the multiway graph.
    Returns a dict of time-series metrics and aggregate statistics.
    """
    metrics = {
        "species": species_name,
        "total_states": MWG.number_of_nodes(),
        "total_transitions": MWG.number_of_edges(),
        "by_layer": defaultdict(dict)
    }

    # Group states by layer (timestep)
    layers = defaultdict(list)
    for node, data in MWG.nodes(data=True):
        layer = data.get('layer', 0)
        layers[layer].append(node)

    max_layer = max(layers.keys()) if layers else 0

    # Time-series metrics for each layer
    for layer in range(max_layer + 1):
        states_at_layer = layers[layer]

        if not states_at_layer:
            continue

        # Count edges in each state (connectivity)
        edge_counts = [len(state) for state in states_at_layer]

        # Survival: states that still have edges
        surviving_states = [s for s in states_at_layer if len(s) > 0]
        dead_states = [s for s in states_at_layer if len(s) == 0]

        # Complexity: unique node count in each state
        node_counts = []
        for state in states_at_layer:
            if state:
                nodes = set()
                for edge in state:
                    nodes.add(edge[0])
                    nodes.add(edge[1])
                node_counts.append(len(nodes))
            else:
                node_counts.append(0)

        metrics["by_layer"][layer] = {
            "num_states": len(states_at_layer),
            "num_surviving": len(surviving_states),
            "num_dead": len(dead_states),
            "survival_rate": len(surviving_states) / len(states_at_layer) if states_at_layer else 0,
            "avg_edges": np.mean(edge_counts) if edge_counts else 0,
            "max_edges": max(edge_counts) if edge_counts else 0,
            "min_edges": min(edge_counts) if edge_counts else 0,
            "std_edges": np.std(edge_counts) if edge_counts else 0,
            "avg_nodes": np.mean(node_counts) if node_counts else 0,
            "max_nodes": max(node_counts) if node_counts else 0
        }

    # Aggregate metrics
    all_edge_counts = []
    all_survival_rates = []
    for layer_data in metrics["by_layer"].values():
        all_edge_counts.append(layer_data["avg_edges"])
        all_survival_rates.append(layer_data["survival_rate"])

    metrics["aggregate"] = {
        "mean_connectivity": np.mean(all_edge_counts) if all_edge_counts else 0,
        "connectivity_decay_rate": (all_edge_counts[0] - all_edge_counts[-1]) / max_layer if max_layer > 0 and all_edge_counts else 0,
        "mean_survival_rate": np.mean(all_survival_rates) if all_survival_rates else 0,
        "final_survival_rate": all_survival_rates[-1] if all_survival_rates else 0,
        "state_space_growth": len(layers[max_layer]) if max_layer in layers else 0
    }

    return metrics

def compare_species_metrics(metrics_glass, metrics_plant):
    """
    Generate comparative analysis between glass and plant systems.
    """
    comparison = {
        "resilience_advantage": {},
        "fragility_indicators": {}
    }

    # Compare aggregate metrics
    agg_g = metrics_glass["aggregate"]
    agg_p = metrics_plant["aggregate"]

    comparison["resilience_advantage"] = {
        "plant_vs_glass_survival": agg_p["final_survival_rate"] - agg_g["final_survival_rate"],
        "plant_vs_glass_connectivity": agg_p["mean_connectivity"] - agg_g["mean_connectivity"],
        "plant_vs_glass_state_space": agg_p["state_space_growth"] - agg_g["state_space_growth"]
    }

    comparison["fragility_indicators"] = {
        "glass_decay_rate": agg_g["connectivity_decay_rate"],
        "plant_decay_rate": agg_p["connectivity_decay_rate"],
        "glass_final_survival": agg_g["final_survival_rate"],
        "plant_final_survival": agg_p["final_survival_rate"]
    }

    return comparison

# ==========================================
# 4. VISUALIZATION
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

def create_evolution_gif(MWG, species_name, output_path, max_layers=None):
    """
    Create an animated GIF showing the temporal evolution of the multiway graph.
    Each frame shows the graph up to a specific layer/timestep.
    """
    print(f"Creating evolution GIF for {species_name}...")

    # Group nodes by layer
    layers = defaultdict(list)
    for node, data in MWG.nodes(data=True):
        layer = data.get('layer', 0)
        layers[layer].append(node)

    if max_layers is None:
        max_layers = max(layers.keys())

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()
    frames = []

    # Compute layout for full graph (so nodes don't move between frames)
    full_pos = get_layout(MWG)

    try:
        for current_layer in range(max_layers + 1):
            # Build subgraph up to current layer
            nodes_up_to_layer = []
            for layer in range(current_layer + 1):
                nodes_up_to_layer.extend(layers[layer])

            subgraph = MWG.subgraph(nodes_up_to_layer).copy()

            if subgraph.number_of_nodes() == 0:
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Use positions from full graph for consistency
            pos = {n: full_pos[n] for n in subgraph.nodes() if n in full_pos}

            # Color nodes by their layer
            node_colors = [MWG.nodes[n].get('layer', 0) for n in subgraph.nodes()]

            # Draw the graph
            nx.draw(subgraph, pos,
                   node_size=80,
                   node_color=node_colors,
                   cmap='viridis',
                   vmin=0,
                   vmax=max_layers,
                   alpha=0.8,
                   edge_color="gray",
                   arrowsize=10,
                   width=0.5,
                   with_labels=False,
                   ax=ax)

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                      norm=plt.Normalize(vmin=0, vmax=max_layers))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Layer/Timestep', rotation=270, labelpad=15)

            # Title with stats
            num_alive = sum(1 for n in subgraph.nodes() if len(n) > 0)
            num_dead = sum(1 for n in subgraph.nodes() if len(n) == 0)

            ax.set_title(f"{species_name.capitalize()} Multiway Evolution\n"
                        f"Layer {current_layer}/{max_layers} | "
                        f"States: {subgraph.number_of_nodes()} | "
                        f"Alive: {num_alive} | Dead: {num_dead}",
                        fontsize=14, fontweight='bold')

            plt.tight_layout()

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{current_layer:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()

            frames.append(imageio.imread(frame_path))

        # Add extra frames at the end (pause on final state)
        for _ in range(5):
            frames.append(frames[-1])

        # Create GIF (3s per frame = 10x slower, infinite loop)
        imageio.mimsave(output_path, frames, duration=3.0, loop=0)
        print(f"  Saved to {output_path}")

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir)

def create_comparison_gif(mwg_glass, mwg_plant, output_path, max_layers=None):
    """
    Create side-by-side comparison GIF of glass and plant evolution.
    """
    print("Creating side-by-side comparison GIF...")

    # Group nodes by layer for both graphs
    layers_glass = defaultdict(list)
    layers_plant = defaultdict(list)

    for node, data in mwg_glass.nodes(data=True):
        layer = data.get('layer', 0)
        layers_glass[layer].append(node)

    for node, data in mwg_plant.nodes(data=True):
        layer = data.get('layer', 0)
        layers_plant[layer].append(node)

    if max_layers is None:
        max_layers = min(max(layers_glass.keys()), max(layers_plant.keys()))

    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    frames = []

    # Compute layouts for full graphs
    full_pos_glass = get_layout(mwg_glass)
    full_pos_plant = get_layout(mwg_plant)

    try:
        for current_layer in range(max_layers + 1):
            # Build subgraphs up to current layer
            nodes_glass = []
            nodes_plant = []

            for layer in range(current_layer + 1):
                nodes_glass.extend(layers_glass[layer])
                nodes_plant.extend(layers_plant[layer])

            subgraph_glass = mwg_glass.subgraph(nodes_glass).copy()
            subgraph_plant = mwg_plant.subgraph(nodes_plant).copy()

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            # --- GLASS SUBPLOT ---
            if subgraph_glass.number_of_nodes() > 0:
                pos_glass = {n: full_pos_glass[n] for n in subgraph_glass.nodes() if n in full_pos_glass}
                node_colors_glass = [mwg_glass.nodes[n].get('layer', 0) for n in subgraph_glass.nodes()]

                nx.draw(subgraph_glass, pos_glass,
                       node_size=60,
                       node_color=SPECIES_DNA["glass"]["color"],
                       alpha=0.7,
                       edge_color="gray",
                       arrowsize=8,
                       width=0.5,
                       with_labels=False,
                       ax=ax1)

                num_alive_glass = sum(1 for n in subgraph_glass.nodes() if len(n) > 0)
                num_dead_glass = sum(1 for n in subgraph_glass.nodes() if len(n) == 0)

                ax1.set_title(f"Glass (Fragile)\n"
                            f"States: {subgraph_glass.number_of_nodes()} | "
                            f"Alive: {num_alive_glass} | Dead: {num_dead_glass}",
                            fontsize=12, fontweight='bold')

            # --- PLANT SUBPLOT ---
            if subgraph_plant.number_of_nodes() > 0:
                pos_plant = {n: full_pos_plant[n] for n in subgraph_plant.nodes() if n in full_pos_plant}
                node_colors_plant = [mwg_plant.nodes[n].get('layer', 0) for n in subgraph_plant.nodes()]

                nx.draw(subgraph_plant, pos_plant,
                       node_size=60,
                       node_color=SPECIES_DNA["plant"]["color"],
                       alpha=0.7,
                       edge_color="gray",
                       arrowsize=8,
                       width=0.5,
                       with_labels=False,
                       ax=ax2)

                num_alive_plant = sum(1 for n in subgraph_plant.nodes() if len(n) > 0)
                num_dead_plant = sum(1 for n in subgraph_plant.nodes() if len(n) == 0)

                ax2.set_title(f"Plant (Resilient)\n"
                            f"States: {subgraph_plant.number_of_nodes()} | "
                            f"Alive: {num_alive_plant} | Dead: {num_dead_plant}",
                            fontsize=12, fontweight='bold')

            # Overall title
            fig.suptitle(f"Multiway Evolution Comparison - Layer {current_layer}/{max_layers}",
                        fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{current_layer:04d}.png")
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            plt.close()

            frames.append(imageio.imread(frame_path))

        # Add pause frames at the end
        for _ in range(5):
            frames.append(frames[-1])

        # Create GIF (1.5s per frame = 5x slower)
        imageio.mimsave(output_path, frames, duration=10.0, loop=0)
        print(f"  Saved to {output_path}")

    finally:
        shutil.rmtree(temp_dir)

def plot_metric_comparison(metrics_glass, metrics_plant):
    """
    Create comprehensive visualization comparing risk metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Extract time series data
    layers_g = sorted([int(k) for k in metrics_glass["by_layer"].keys()])
    layers_p = sorted([int(k) for k in metrics_plant["by_layer"].keys()])

    # 1. Survival Rate over Time
    ax = axes[0, 0]
    survival_g = [metrics_glass["by_layer"][l]["survival_rate"] for l in layers_g]
    survival_p = [metrics_plant["by_layer"][l]["survival_rate"] for l in layers_p]
    ax.plot(layers_g, survival_g, 'o-', color=SPECIES_DNA["glass"]["color"], label="Glass", linewidth=2)
    ax.plot(layers_p, survival_p, 's-', color=SPECIES_DNA["plant"]["color"], label="Plant", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Survival Rate")
    ax.set_title("Survival Rate Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Average Connectivity over Time
    ax = axes[0, 1]
    conn_g = [metrics_glass["by_layer"][l]["avg_edges"] for l in layers_g]
    conn_p = [metrics_plant["by_layer"][l]["avg_edges"] for l in layers_p]
    ax.plot(layers_g, conn_g, 'o-', color=SPECIES_DNA["glass"]["color"], label="Glass", linewidth=2)
    ax.plot(layers_p, conn_p, 's-', color=SPECIES_DNA["plant"]["color"], label="Plant", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average Edges per State")
    ax.set_title("Connectivity Decay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. State Space Growth
    ax = axes[0, 2]
    states_g = [metrics_glass["by_layer"][l]["num_states"] for l in layers_g]
    states_p = [metrics_plant["by_layer"][l]["num_states"] for l in layers_p]
    ax.plot(layers_g, states_g, 'o-', color=SPECIES_DNA["glass"]["color"], label="Glass", linewidth=2)
    ax.plot(layers_p, states_p, 's-', color=SPECIES_DNA["plant"]["color"], label="Plant", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of States")
    ax.set_title("State Space Exploration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Dead States Accumulation
    ax = axes[1, 0]
    dead_g = [metrics_glass["by_layer"][l]["num_dead"] for l in layers_g]
    dead_p = [metrics_plant["by_layer"][l]["num_dead"] for l in layers_p]
    ax.plot(layers_g, dead_g, 'o-', color=SPECIES_DNA["glass"]["color"], label="Glass", linewidth=2)
    ax.plot(layers_p, dead_p, 's-', color=SPECIES_DNA["plant"]["color"], label="Plant", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Dead States (Empty Graphs)")
    ax.set_title("Failure Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Connectivity Distribution (Box plot at final layer)
    ax = axes[1, 1]
    final_layer_g = max(layers_g)
    final_layer_p = max(layers_p)

    # Get edge counts for all states at final layer
    final_states_g = [len(s) for s in metrics_glass["by_layer"][final_layer_g].get("_states", [])]
    final_states_p = [len(s) for s in metrics_plant["by_layer"][final_layer_p].get("_states", [])]

    # Bar comparison of aggregate metrics
    categories = ['Mean\nConnectivity', 'Final\nSurvival', 'Decay\nRate']
    glass_vals = [
        metrics_glass["aggregate"]["mean_connectivity"],
        metrics_glass["aggregate"]["final_survival_rate"],
        metrics_glass["aggregate"]["connectivity_decay_rate"]
    ]
    plant_vals = [
        metrics_plant["aggregate"]["mean_connectivity"],
        metrics_plant["aggregate"]["final_survival_rate"],
        metrics_plant["aggregate"]["connectivity_decay_rate"]
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, glass_vals, width, label='Glass', color=SPECIES_DNA["glass"]["color"], alpha=0.8)
    ax.bar(x + width/2, plant_vals, width, label='Plant', color=SPECIES_DNA["plant"]["color"], alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('Aggregate Risk Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Summary Table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ["Metric", "Glass", "Plant", "Δ"],
        ["Total States",
         f"{metrics_glass['total_states']}",
         f"{metrics_plant['total_states']}",
         f"+{metrics_plant['total_states'] - metrics_glass['total_states']}"],
        ["Final Survival",
         f"{metrics_glass['aggregate']['final_survival_rate']:.2%}",
         f"{metrics_plant['aggregate']['final_survival_rate']:.2%}",
         f"+{(metrics_plant['aggregate']['final_survival_rate'] - metrics_glass['aggregate']['final_survival_rate'])*100:.1f}%"],
        ["Mean Connect.",
         f"{metrics_glass['aggregate']['mean_connectivity']:.2f}",
         f"{metrics_plant['aggregate']['mean_connectivity']:.2f}",
         f"+{metrics_plant['aggregate']['mean_connectivity'] - metrics_glass['aggregate']['mean_connectivity']:.2f}"],
        ["Decay Rate",
         f"{metrics_glass['aggregate']['connectivity_decay_rate']:.3f}",
         f"{metrics_plant['aggregate']['connectivity_decay_rate']:.3f}",
         f"{metrics_plant['aggregate']['connectivity_decay_rate'] - metrics_glass['aggregate']['connectivity_decay_rate']:.3f}"]
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle("Risk Metrics Comparison: Glass vs Plant Resilience", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("risk_metrics_comparison.png", dpi=150, bbox_inches='tight')
    print("Risk metrics plot saved to risk_metrics_comparison.png")

def plot_futures(steps):
    # Generate the graphs
    print("Generating Glass Futures...")
    mwg_glass = generate_multiway_graph("glass", steps=steps)
    print(f"  Glass: {mwg_glass.number_of_nodes()} states, {mwg_glass.number_of_edges()} transitions")

    print("Generating Plant Futures...")
    mwg_plant = generate_multiway_graph("plant", steps=steps)
    print(f"  Plant: {mwg_plant.number_of_nodes()} states, {mwg_plant.number_of_edges()} transitions")

    # Compute risk metrics
    print("\nComputing risk metrics...")
    metrics_glass = compute_risk_metrics(mwg_glass, "glass")
    metrics_plant = compute_risk_metrics(mwg_plant, "plant")
    comparison = compare_species_metrics(metrics_glass, metrics_plant)

    # Save metrics to JSON
    print("Saving metrics to JSON...")
    with open("risk_metrics.json", "w") as f:
        json.dump({
            "glass": metrics_glass,
            "plant": metrics_plant,
            "comparison": comparison
        }, f, indent=2, default=str)
    print("Metrics saved to risk_metrics.json")

    # Plot multiway graphs
    plt.figure(figsize=(14, 7))

    # --- PLOT GLASS FUTURES ---
    plt.subplot(1, 2, 1)
    pos_glass = get_layout(mwg_glass)

    nx.draw(mwg_glass, pos_glass, node_size=50, node_color=SPECIES_DNA["glass"]["color"],
            alpha=0.7, edge_color="gray", arrowsize=15)
    plt.title(f"Glass Multiway Graph\nNodes: {mwg_glass.number_of_nodes()} | Edges: {mwg_glass.number_of_edges()}\nFinal Survival: {metrics_glass['aggregate']['final_survival_rate']:.1%}")

    # --- PLOT PLANT FUTURES ---
    plt.subplot(1, 2, 2)
    pos_plant = get_layout(mwg_plant)

    nx.draw(mwg_plant, pos_plant, node_size=50, node_color=SPECIES_DNA["plant"]["color"],
            alpha=0.7, edge_color="gray", arrowsize=15)
    plt.title(f"Plant Multiway Graph\nNodes: {mwg_plant.number_of_nodes()} | Edges: {mwg_plant.number_of_edges()}\nFinal Survival: {metrics_plant['aggregate']['final_survival_rate']:.1%}")

    plt.tight_layout()
    plt.savefig("multiway_futures_comparison.png", dpi=150)
    print("Multiway plot saved to multiway_futures_comparison.png")

    # Plot risk metrics comparison
    print("\nGenerating risk metrics visualizations...")
    plot_metric_comparison(metrics_glass, metrics_plant)

    # Create evolution GIFs
    print("\nGenerating temporal evolution GIFs...")
    create_evolution_gif(mwg_glass, "glass", "glass_evolution.gif", max_layers=steps)
    create_evolution_gif(mwg_plant, "plant", "plant_evolution.gif", max_layers=steps)
    create_comparison_gif(mwg_glass, mwg_plant, "comparison_evolution.gif", max_layers=steps)

    print("\n" + "="*60)
    print("RISK ANALYSIS SUMMARY")
    print("="*60)
    print(f"Plant Resilience Advantage (vs Glass):")
    print(f"  • Survival Rate: +{comparison['resilience_advantage']['plant_vs_glass_survival']*100:.1f}%")
    print(f"  • Mean Connectivity: +{comparison['resilience_advantage']['plant_vs_glass_connectivity']:.2f} edges")
    print(f"  • State Space: +{comparison['resilience_advantage']['plant_vs_glass_state_space']} states")
    print(f"\nFragility Indicators:")
    print(f"  • Glass Decay Rate: {comparison['fragility_indicators']['glass_decay_rate']:.3f} edges/step")
    print(f"  • Plant Decay Rate: {comparison['fragility_indicators']['plant_decay_rate']:.3f} edges/step")
    print("="*60)

    plt.show()

if __name__ == "__main__":
    plot_futures(10)
