"""
Topological Risk Analysis: Multiway Causal Graph Framework
===========================================================

This code implements a purely topological approach to risk analysis for 
Social-Ecological-Technological Systems (SETS). It generates multiway causal 
graphs through local graph rewriting rules and quantifies risk through 
topological metrics without probabilistic forecasting.

Key metrics:
- N_paths: Pathway multiplicity (raw count of distinct causal pathways)
- ρ (rho): Reconvergence coefficient (structural degeneracy)
- S(θ): Topological sensitivity to parameter changes
- λ₂ (lambda_2): Spectral gap (algebraic connectivity)

Author: [Your Name]
Date: 2025
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.linalg


# ==========================================
# GENERATIVE GRAMMAR: GRAPH REWRITING RULES
# ==========================================

def get_rule_signature(rule_type):
    """
    Returns the observables (ΔU, ΔE, ΔC) consequent to graph rewriting operations.
    
    These values are NOT determinants of the rules but emergent properties 
    measured after the rule is applied to a configuration.
    
    Parameters:
    -----------
    rule_type : str
        Either 'entropy' (degradation/deletion) or 'repair' (creation)
    
    Returns:
    --------
    dict : Contains ΔU (utility), ΔE (effort), ΔC (constraint), and j (action)
    """
    if rule_type == "entropy": 
        # Edge Deletion: U↓, E↓, C↑ (system degrades, loses options)
        return {"dU": -1, "dE": -1, "dC": 1, "j": 1}
    
    elif rule_type == "repair":
        # Edge Creation: U↑, E↑, C↓ (system adapts, gains options)
        return {"dU": 1, "dE": 1, "dC": -1, "j": -1}
    
    return {"j": 0}


def generate_topology(species_type, steps=8):
    """
    Generates the Multiway Causal Graph through iterative rule application.
    
    The multiway graph represents ALL possible futures, where:
    - Nodes = system configurations (frozen sets of edges)
    - Edges = rule applications (graph rewriting operations)
    
    Parameters:
    -----------
    species_type : str
        'glass' (degradation only) or 'plant' (degradation + repair)
    steps : int
        Number of temporal layers to generate
    
    Returns:
    --------
    nx.DiGraph : The multiway causal graph
    """
    # Initial configuration: triangle (ring of 3 nodes)
    initial_state = frozenset({(1,2), (2,3), (3,1)})
    
    MWG = nx.DiGraph()
    MWG.add_node(initial_state, layer=0)
    
    frontier = [initial_state]
    seen = {initial_state}
    
    for step in range(1, steps + 1):
        new_frontier = []
        
        for state in frontier:
            # Empty state has no futures
            if not state: 
                continue
            
            # RULE 1: ENTROPY (Universal - applies to all systems)
            # Remove each edge one at a time
            for edge in state:
                next_edges = set(state)
                next_edges.remove(edge)
                next_state = frozenset(next_edges)
                
                sig = get_rule_signature("entropy")
                MWG.add_edge(state, next_state, cost=sig['j'], type='entropy')
                
                if next_state not in seen:
                    MWG.add_node(next_state, layer=step)
                    seen.add(next_state)
                    new_frontier.append(next_state)

            # RULE 2: REPAIR (Plant only - adaptive systems)
            # Close open triangles through transitivity
            if species_type == "plant":
                G_temp = nx.Graph(list(state))
                nodes = list(G_temp.nodes())
                
                possible_repairs = []
                for n in nodes:
                    neighbors = list(G_temp.neighbors(n))
                    for n1, n2 in combinations(neighbors, 2):
                        if not G_temp.has_edge(n1, n2):
                            # Found open triangle: n1-n-n2
                            repair_edge = tuple(sorted((n1, n2)))
                            possible_repairs.append(repair_edge)
                
                for rep_edge in possible_repairs:
                    next_edges = set(state)
                    next_edges.add(rep_edge)
                    next_state = frozenset(next_edges)
                    
                    sig = get_rule_signature("repair")
                    MWG.add_edge(state, next_state, cost=sig['j'], type='repair')
                    
                    if next_state not in seen:
                        MWG.add_node(next_state, layer=step)
                        seen.add(next_state)
                        new_frontier.append(next_state)
                        
        frontier = new_frontier
    
    return MWG


# ==========================================
# TOPOLOGICAL RISK METRICS
# ==========================================

def analyze_pathway_topology(MWG):
    """
    Computes purely topological metrics on the multiway causal graph.
    
    No probability distributions are assigned - metrics measure the 
    structure of the possibility space directly.
    
    Returns:
    --------
    tuple : (N_paths, rho, lambda_2)
        - N_paths: Number of distinct pathways
        - rho: Reconvergence coefficient
        - lambda_2: Spectral gap
    """
    # Find root (initial state at layer 0)
    root = [n for n, attr in MWG.nodes(data=True) if attr.get('layer', -1) == 0][0]

    # Find terminal nodes (leaves or max layer nodes)
    leaves = [n for n, d in MWG.out_degree() if d == 0]
    if not leaves:
        max_layer = max(attr.get('layer', 0) for _, attr in MWG.nodes(data=True))
        leaves = [n for n, attr in MWG.nodes(data=True) if attr.get('layer', -1) == max_layer]
    
    # METRIC 1: Pathway Multiplicity (N_paths)
    # Direct count of distinct pathways through the multiway graph
    paths = []
    max_paths_per_leaf = 10000  # Prevent combinatorial explosion
    
    for leaf in leaves:
        try:
            path_generator = nx.all_simple_paths(MWG, root, leaf, cutoff=20)
            leaf_paths = []
            for path in path_generator:
                paths.append(path)
                leaf_paths.append(path)
                if len(leaf_paths) >= max_paths_per_leaf:
                    break
        except nx.NetworkXNoPath:
            continue
    
    n_paths = len(paths)
    
    # METRIC 2: Reconvergence Coefficient (ρ)
    # Fraction of nodes with ≥2 incoming edges (structural degeneracy)
    nodes_with_reconvergence = 0
    total_nodes = MWG.number_of_nodes()
    
    for node in MWG.nodes():
        if MWG.in_degree(node) >= 2:
            nodes_with_reconvergence += 1
    
    rho = nodes_with_reconvergence / total_nodes if total_nodes > 0 else 0
    
    # METRIC 3: Spectral Gap (λ₂)
    # Algebraic connectivity via Laplacian's second eigenvalue
    undir_G = MWG.to_undirected()
    largest_cc = max(nx.connected_components(undir_G), key=len)
    subgraph = undir_G.subgraph(largest_cc)
    
    if len(subgraph) > 1:
        L = nx.normalized_laplacian_matrix(subgraph)
        eigenvalues = scipy.linalg.eigvalsh(L.todense())
        eigenvalues.sort()
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
    else:
        spectral_gap = 0
    
    return n_paths, rho, spectral_gap


def compute_topological_sensitivity(species_type, base_steps=6, perturbation_steps=7):
    """
    Computes topological sensitivity S(θ) as the rate of change of 
    topological properties with respect to a parameter θ.
    
    Here θ = simulation depth (number of steps).
    
    Returns:
    --------
    float : Sensitivity measure (sum of normalized metric changes)
    """
    # Generate at base parameter value
    mwg_base = generate_topology(species_type, steps=base_steps)
    n_base, rho_base, lambda_base = analyze_pathway_topology(mwg_base)
    
    # Generate at perturbed parameter value
    mwg_pert = generate_topology(species_type, steps=perturbation_steps)
    n_pert, rho_pert, lambda_pert = analyze_pathway_topology(mwg_pert)
    
    # Compute sensitivity as absolute change per unit parameter change
    delta_steps = perturbation_steps - base_steps
    
    sensitivity = (
        abs(n_pert - n_base) / delta_steps +
        abs(rho_pert - rho_base) / delta_steps +
        abs(lambda_pert - lambda_base) / delta_steps
    )
    
    return sensitivity


# ==========================================
# COMPARATIVE ANALYSIS & VISUALIZATION
# ==========================================

def run_analysis():
    """
    Main analysis function: generates Glass and Plant systems, 
    computes topological metrics, and produces visualizations.
    """
    print("=" * 60)
    print("TOPOLOGICAL RISK ANALYSIS: GLASS VS PLANT")
    print("=" * 60)
    
    print("\nGenerating Multiway Causal Graphs...")
    steps = 6
    mwg_glass = generate_topology("glass", steps=steps)
    mwg_plant = generate_topology("plant", steps=steps)
    
    print(f"  Glass system: {mwg_glass.number_of_nodes()} configurations")
    print(f"  Plant system: {mwg_plant.number_of_nodes()} configurations")

    # Compute topological metrics
    print("\nComputing topological metrics...")
    glass_n, glass_rho, glass_lambda = analyze_pathway_topology(mwg_glass)
    plant_n, plant_rho, plant_lambda = analyze_pathway_topology(mwg_plant)
    
    print(f"\n  Glass: N_paths={glass_n}, ρ={glass_rho:.3f}, λ₂={glass_lambda:.4f}")
    print(f"  Plant: N_paths={plant_n}, ρ={plant_rho:.3f}, λ₂={plant_lambda:.4f}")
    
    # Compute topological sensitivity
    print("\nComputing topological sensitivity S(θ)...")
    glass_sensitivity = compute_topological_sensitivity("glass", base_steps=5, perturbation_steps=6)
    plant_sensitivity = compute_topological_sensitivity("plant", base_steps=5, perturbation_steps=6)
    
    print(f"  Glass S(θ): {glass_sensitivity:.3f}")
    print(f"  Plant S(θ): {plant_sensitivity:.3f}")
    
    # Temporal evolution analysis
    print("\nAnalyzing temporal evolution...")
    step_range = range(3, 8)
    glass_evolution_n = []
    plant_evolution_n = []
    glass_evolution_rho = []
    plant_evolution_rho = []
    
    for s in step_range:
        g_mwg = generate_topology("glass", steps=s)
        p_mwg = generate_topology("plant", steps=s)
        
        g_n, g_rho, _ = analyze_pathway_topology(g_mwg)
        p_n, p_rho, _ = analyze_pathway_topology(p_mwg)
        
        glass_evolution_n.append(g_n)
        plant_evolution_n.append(p_n)
        glass_evolution_rho.append(g_rho)
        plant_evolution_rho.append(p_rho)

    # Create comparison table
    print("\nGenerating comparison table...")
    fig_table = plt.figure(figsize=(12, 8))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('tight')
    ax_table.axis('off')

    table_data = [
        ["Metric", "Glass", "Plant", "Interpretation"],
        ["Total States",
         f"{mwg_glass.number_of_nodes()}",
         f"{mwg_plant.number_of_nodes()}",
         "Configuration space size"],
        ["N_paths",
         f"{glass_n}",
         f"{plant_n}",
         "Pathway multiplicity"],
        ["ρ (Reconvergence)",
         f"{glass_rho:.3f}",
         f"{plant_rho:.3f}",
         "Structural degeneracy"],
        ["S(θ) (Sensitivity)",
         f"{glass_sensitivity:.3f}",
         f"{plant_sensitivity:.3f}",
         "Topological transition risk"],
        ["λ₂ (Spectral Gap)",
         f"{glass_lambda:.4f}",
         f"{plant_lambda:.4f}",
         "Algebraic connectivity"],
    ]

    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.30, 0.20, 0.20, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

    # Alternate row colors
    for i in range(1, len(table_data)):
        if i % 2 == 1:
            for j in range(4):
                table[(i, j)].set_facecolor('#f5f5f5')

    plt.title("Topological Risk Metrics: Glass vs Plant Regimes", 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig("topological_risk_table.png", dpi=300, bbox_inches='tight')
    print("  ✓ Table saved: topological_risk_table.png")

    # Create visualization plots
    print("\nGenerating visualization plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Pathway Count Evolution
    ax = axes[0, 0]
    ax.plot(step_range, glass_evolution_n, 'o-', color='#e74c3c', 
            label='Glass (Degradation)', linewidth=2.5, markersize=8)
    ax.plot(step_range, plant_evolution_n, 's-', color='#27ae60', 
            label='Plant (Adaptive)', linewidth=2.5, markersize=8)
    ax.set_title("Pathway Multiplicity (N_paths)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Simulation Depth (Steps)", fontsize=11)
    ax.set_ylabel("Number of Distinct Pathways", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Reconvergence Evolution
    ax = axes[0, 1]
    ax.plot(step_range, glass_evolution_rho, 'o-', color='#e74c3c', 
            label='Glass', linewidth=2.5, markersize=8)
    ax.plot(step_range, plant_evolution_rho, 's-', color='#27ae60', 
            label='Plant', linewidth=2.5, markersize=8)
    ax.set_title("Reconvergence Coefficient (ρ)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Simulation Depth (Steps)", fontsize=11)
    ax.set_ylabel("Fraction of Nodes with Reconvergence", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Topological Sensitivity
    ax = axes[1, 0]
    categories = ['Glass', 'Plant']
    sensitivities = [glass_sensitivity, plant_sensitivity]
    bars = ax.bar(categories, sensitivities, color=['#e74c3c', '#27ae60'], 
                   alpha=0.8, width=0.5, edgecolor='black', linewidth=1.5)
    ax.set_title("Topological Sensitivity S(θ)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Sensitivity to Parameter Changes", fontsize=11)
    if max(sensitivities) > 0:
        ax.set_ylim(0, max(sensitivities)*1.3)
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 4: Spectral Gap
    ax = axes[1, 1]
    values = [glass_lambda, plant_lambda]
    bars = ax.bar(categories, values, color=['#e74c3c', '#27ae60'], 
                   alpha=0.8, width=0.5, edgecolor='black', linewidth=1.5)
    ax.set_title("Algebraic Connectivity (λ₂)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Spectral Gap", fontsize=11)
    ax.set_ylim(0, max(values)*1.3)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.text(0.5, max(values)*1.15, 
            "λ₂ → 0 indicates topological collapse",
            ha='center', fontsize=9, style='italic', color='red')

    plt.tight_layout()
    plt.savefig("topological_risk_metrics.png", dpi=300, bbox_inches='tight')
    print("  ✓ Plots saved: topological_risk_metrics.png")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    run_analysis()