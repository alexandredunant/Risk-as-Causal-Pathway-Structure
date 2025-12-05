"""
Italy Demographic Trajectory: Emergent Eras from Unified Topology
===================================================================

All transformation rules operate simultaneously without era-specific switches.
Different demographic regimes (growth, boom, decline) emerge as distinct regions
in the multiway causal graph based on the intrinsic topology of rule interactions.

The system discovers its own transitions rather than having them imposed externally.

Author: [Your Name]
Date: 2025
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.linalg
from collections import defaultdict


# ==========================================
# UNIVERSAL TRANSFORMATION RULES
# ==========================================

def get_universal_rules():
    """
    Returns ALL transformation rules that can operate on demographic systems.
    These rules are always available - no era switching.
    Different demographic regimes emerge from which rules successfully apply
    to which configurations, not from external selection.
    """
    rules = []
    
    # RULE 1: FERTILITY DECLINE
    # Pattern: Any edge exists
    # Transformation: Remove edge (connection breaks)
    def fertility_check(state):
        return len(state) > 0
    
    def fertility_transform(state):
        successors = []
        for edge in state:
            next_state = frozenset(set(state) - {edge})
            successors.append(('fertility_decline', next_state))
        return successors
    
    rules.append(('fertility_decline', fertility_check, fertility_transform))
    
    # RULE 2: TECHNOLOGY EXPANSION (Infrastructure)
    # Pattern: Open triangle exists
    # Transformation: Close triangle (create connection)
    def tech_check(state):
        if len(state) < 2:
            return False
        G = nx.Graph(list(state))
        nodes = list(G.nodes())
        if len(nodes) < 3:
            return False
        for n1, n2 in combinations(nodes, 2):
            if not G.has_edge(n1, n2):
                if set(G.neighbors(n1)) & set(G.neighbors(n2)):
                    return True
        return False
    
    def tech_transform(state):
        successors = []
        G = nx.Graph(list(state))
        nodes = list(G.nodes())
        
        for n1, n2 in combinations(nodes, 2):
            if not G.has_edge(n1, n2):
                if set(G.neighbors(n1)) & set(G.neighbors(n2)):
                    new_edge = tuple(sorted((n1, n2)))
                    next_state = frozenset(set(state) | {new_edge})
                    successors.append(('technology', next_state))
                    if len(successors) >= 3:  # Limit to prevent explosion
                        break
        return successors
    
    rules.append(('technology', tech_check, tech_transform))
    
    # RULE 3: SYSTEMIC COLLAPSE (Cascade failure)
    # Pattern: Multiple edges exist
    # Transformation: Remove connected edges simultaneously
    def collapse_check(state):
        return len(state) >= 3
    
    def collapse_transform(state):
        successors = []
        edges = list(state)
        
        if len(edges) >= 3:
            # Find edges that form connected components
            for i, e1 in enumerate(edges):
                for j, e2 in enumerate(edges[i+1:], i+1):
                    # If edges share a node (connected)
                    if e1[0] in e2 or e1[1] in e2:
                        next_state = frozenset(set(state) - {e1, e2})
                        successors.append(('collapse', next_state))
                        if len(successors) >= 2:
                            return successors
        return successors
    
    rules.append(('collapse', collapse_check, collapse_transform))
    
    return rules


# ==========================================
# UNIFIED MULTIWAY GRAPH GENERATION
# ==========================================

def generate_unified_topology(max_steps=10, max_states=5000):
    """
    Generates a single unified multiway causal graph where ALL rules
    operate simultaneously. Different demographic regimes emerge as
    distinct topological regions rather than being prescribed beforehand.
    
    Parameters:
    -----------
    max_steps : int
        Maximum temporal depth
    max_states : int
        Maximum states to prevent combinatorial explosion
    
    Returns:
    --------
    nx.DiGraph : The unified multiway causal graph
    """
    # Initial configuration: Pentagon (5 demographic capacity units)
    initial_state = frozenset({(1,2), (2,3), (3,4), (4,5), (5,1)})
    
    rules = get_universal_rules()
    
    MWG = nx.DiGraph()
    MWG.add_node(initial_state, layer=0, size=len(initial_state))
    
    frontier = [initial_state]
    seen = {initial_state}
    
    for step in range(1, max_steps + 1):
        if len(seen) >= max_states:
            print(f"  Reached max_states limit at step {step}")
            break
        
        new_frontier = []
        
        for state in frontier:
            if not state:
                continue
            
            # Apply ALL rules that match this state
            for rule_name, check_fn, transform_fn in rules:
                if check_fn(state):
                    successors = transform_fn(state)
                    
                    for rule_used, next_state in successors:
                        # Record which rule generated this transition
                        MWG.add_edge(state, next_state, rule=rule_used)
                        
                        if next_state not in seen:
                            MWG.add_node(next_state, layer=step, size=len(next_state))
                            seen.add(next_state)
                            new_frontier.append(next_state)
        
        frontier = new_frontier
        print(f"  Step {step}: {len(frontier)} new states, {len(seen)} total")
    
    return MWG


# ==========================================
# REGIME IDENTIFICATION FROM TOPOLOGY
# ==========================================

def identify_regimes(MWG):
    """
    Analyzes the unified multiway graph to identify distinct demographic
    regimes based on topological properties. Regimes emerge from the
    structure rather than being prescribed.
    
    Classification based on:
    - Size trajectory: growing, stable, declining configurations
    - Rule dominance: which rules are primarily active
    - Connectivity: pathway diversity and reconvergence patterns
    """
    regimes = {
        'growth': [],      # Configurations with increasing edges
        'stable': [],      # Configurations maintaining edges  
        'decline': [],     # Configurations with decreasing edges
        'collapse': []     # Configurations with catastrophic loss
    }
    
    for node in MWG.nodes():
        size = len(node)
        
        # Analyze successors to classify regime
        successors = list(MWG.successors(node))
        if not successors:
            if size == 0:
                regimes['collapse'].append(node)
            else:
                regimes['stable'].append(node)
            continue
        
        # Count rule types in outgoing edges
        rule_counts = defaultdict(int)
        for succ in successors:
            rule = MWG[node][succ].get('rule', 'unknown')
            rule_counts[rule] += 1
        
        # Classify based on dominant rule pattern
        if rule_counts['technology'] > rule_counts['fertility_decline']:
            regimes['growth'].append(node)
        elif rule_counts['collapse'] > 0:
            regimes['decline'].append(node)
        elif size <= 2:
            regimes['collapse'].append(node)
        else:
            regimes['stable'].append(node)
    
    return regimes


def analyze_regime_topology(MWG, regime_nodes):
    """
    Computes topological metrics for a specific regime (subgraph).
    """
    if not regime_nodes:
        return 0, 0.0, 0.0
    
    subgraph = MWG.subgraph(regime_nodes)
    
    # Find roots and leaves in this regime
    roots = [n for n in regime_nodes if subgraph.in_degree(n) == 0]
    leaves = [n for n in regime_nodes if subgraph.out_degree(n) == 0]
    
    if not roots or not leaves:
        return len(regime_nodes), 0.0, 0.0
    
    # Count pathways
    n_paths = 0
    for root in roots[:5]:  # Limit to prevent explosion
        for leaf in leaves[:5]:
            try:
                paths = list(nx.all_simple_paths(subgraph, root, leaf, cutoff=10))
                n_paths += len(paths)
                if n_paths > 1000:
                    break
            except:
                continue
        if n_paths > 1000:
            break
    
    # Reconvergence
    reconvergent = sum(1 for n in regime_nodes if subgraph.in_degree(n) >= 2)
    rho = reconvergent / len(regime_nodes) if regime_nodes else 0
    
    # Spectral gap
    undir = subgraph.to_undirected()
    if len(undir) > 1:
        try:
            largest_cc = max(nx.connected_components(undir), key=len)
            cc_graph = undir.subgraph(largest_cc)
            if len(cc_graph) > 1:
                L = nx.normalized_laplacian_matrix(cc_graph)
                eigs = scipy.linalg.eigvalsh(L.todense())
                eigs.sort()
                lambda_2 = eigs[1] if len(eigs) > 1 else 0
            else:
                lambda_2 = 0
        except:
            lambda_2 = 0
    else:
        lambda_2 = 0
    
    return n_paths, rho, lambda_2


# ==========================================
# VISUALIZATION
# ==========================================

def visualize_unified_graph(MWG, regimes, max_display=150):
    """
    Visualizes the unified multiway graph with regimes color-coded.
    """
    # Sample nodes if too large
    if MWG.number_of_nodes() > max_display:
        # Sample from each regime proportionally
        display_nodes = []
        for regime_name, nodes in regimes.items():
            sample_size = min(len(nodes), max_display // 4)
            display_nodes.extend(list(nodes)[:sample_size])
        MWG_display = MWG.subgraph(display_nodes).copy()
    else:
        MWG_display = MWG
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    
    # Hierarchical layout by layers
    pos = {}
    layers = defaultdict(list)
    for node, attr in MWG_display.nodes(data=True):
        layer = attr.get('layer', 0)
        layers[layer].append(node)
    
    for layer, nodes in layers.items():
        y = -layer
        x_spacing = 2.0
        x_offset = -(len(nodes) - 1) * x_spacing / 2
        for i, node in enumerate(nodes):
            pos[node] = (x_offset + i * x_spacing, y)
    
    # Color nodes by regime
    node_colors = []
    for node in MWG_display.nodes():
        if node in regimes['growth']:
            node_colors.append('#27ae60')  # Green for growth
        elif node in regimes['stable']:
            node_colors.append('#3498db')  # Blue for stable
        elif node in regimes['decline']:
            node_colors.append('#f39c12')  # Orange for decline
        elif node in regimes['collapse']:
            node_colors.append('#e74c3c')  # Red for collapse
        else:
            node_colors.append('#95a5a6')  # Gray for unknown
    
    # Draw nodes
    nx.draw_networkx_nodes(MWG_display, pos, node_color=node_colors,
                          node_size=200, alpha=0.7, ax=ax)
    
    # Draw edges by rule type
    for u, v, data in MWG_display.edges(data=True):
        rule = data.get('rule', 'unknown')
        if rule == 'fertility_decline':
            color, style = '#e74c3c', 'solid'
        elif rule == 'technology':
            color, style = '#27ae60', 'solid'
        elif rule == 'collapse':
            color, style = '#8e44ad', 'dashed'
        else:
            color, style = '#95a5a6', 'dotted'
        
        nx.draw_networkx_edges(MWG_display, pos, [(u, v)],
                              edge_color=color, width=1.5, alpha=0.4,
                              arrows=True, arrowsize=8, style=style, ax=ax)
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Growth Regime'),
        Patch(facecolor='#3498db', label='Stable Regime'),
        Patch(facecolor='#f39c12', label='Decline Regime'),
        Patch(facecolor='#e74c3c', label='Collapse Regime'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label='Fertility Decline'),
        Line2D([0], [0], color='#27ae60', linewidth=2, label='Technology'),
        Line2D([0], [0], color='#8e44ad', linewidth=2, linestyle='--', label='Collapse'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title(f"Unified Multiway Graph: Emergent Demographic Regimes\n" +
                f"({MWG.number_of_nodes()} states, {MWG.number_of_edges()} transitions)",
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ==========================================
# MAIN ANALYSIS
# ==========================================

def run_unified_analysis():
    """
    Generates a single unified multiway graph and discovers demographic
    regimes from the emergent topology.
    """
    print("=" * 70)
    print("ITALY DEMOGRAPHIC TRAJECTORY: UNIFIED TOPOLOGICAL ANALYSIS")
    print("=" * 70)
    print("\nAll transformation rules operate simultaneously.")
    print("Demographic regimes emerge from topology, not from prescription.\n")
    
    print("Generating unified multiway causal graph...")
    MWG = generate_unified_topology(max_steps=6, max_states=800)
    
    print(f"\nGenerated graph:")
    print(f"  Total configurations: {MWG.number_of_nodes()}")
    print(f"  Total transitions: {MWG.number_of_edges()}")
    
    # Identify emergent regimes
    print("\nIdentifying emergent demographic regimes from topology...")
    regimes = identify_regimes(MWG)
    
    print(f"\nDiscovered regimes:")
    for regime_name, nodes in regimes.items():
        print(f"  {regime_name.capitalize()}: {len(nodes)} configurations")
    
    # Analyze each regime
    print("\n" + "=" * 70)
    print("REGIME TOPOLOGICAL CHARACTERISTICS")
    print("=" * 70)
    
    regime_metrics = {}
    for regime_name, nodes in regimes.items():
        if nodes:
            n_paths, rho, lambda_2 = analyze_regime_topology(MWG, nodes)
            regime_metrics[regime_name] = {
                'configs': len(nodes),
                'n_paths': n_paths,
                'rho': rho,
                'lambda_2': lambda_2
            }
            print(f"\n{regime_name.capitalize()} Regime:")
            print(f"  Configurations: {len(nodes)}")
            print(f"  Pathways: {n_paths}")
            print(f"  Reconvergence (ρ): {rho:.3f}")
            print(f"  Spectral gap (λ₂): {lambda_2:.4f}")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = visualize_unified_graph(MWG, regimes)
    plt.savefig("italy_unified_topology.png", dpi=300, bbox_inches='tight')
    print("  ✓ Graph saved: italy_unified_topology.png")
    
    # Create metrics table
    fig_table = plt.figure(figsize=(12, 6))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = [["Regime", "Configurations", "Pathways", "ρ", "λ₂"]]
    for regime_name in ['growth', 'stable', 'decline', 'collapse']:
        if regime_name in regime_metrics:
            m = regime_metrics[regime_name]
            table_data.append([
                regime_name.capitalize(),
                f"{m['configs']}",
                f"{m['n_paths']}",
                f"{m['rho']:.3f}",
                f"{m['lambda_2']:.4f}"
            ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.25, 0.20, 0.20, 0.17, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    colors = {'growth': '#ccffcc', 'stable': '#cce5ff', 
              'decline': '#ffe5cc', 'collapse': '#ffcccc'}
    for i in range(1, len(table_data)):
        regime = table_data[i][0].lower()
        color = colors.get(regime, '#f5f5f5')
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    plt.title("Emergent Demographic Regimes: Topological Characteristics",
              fontsize=13, fontweight='bold', pad=20)
    plt.savefig("italy_unified_regimes_table.png", dpi=300, bbox_inches='tight')
    print("  ✓ Table saved: italy_unified_regimes_table.png")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Insight: Demographic transitions emerge from the topology")
    print("of rule interactions rather than from external era definitions.")
    print("The same rule set generates growth, stability, and decline regimes")
    print("depending on which configurations the system occupies.")
    
    plt.show()


if __name__ == "__main__":
    run_unified_analysis()