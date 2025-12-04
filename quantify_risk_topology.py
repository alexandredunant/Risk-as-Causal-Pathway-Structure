import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import scipy.linalg

# ==========================================
# 1. GENERATIVE GRAMMAR (RULES & ACTION)
# ==========================================

def get_rule_signature(rule_type):
    """
    Returns the (Delta_U, Delta_E, Delta_C) signature and net Action cost 'j'.
    Action j = Delta_E - Delta_U + Delta_C
    """
    if rule_type == "entropy": 
        # Edge Deletion: U decreases (-1), E decreases (-1), C increases (+1)
        # j = -1 - (-1) + 1 = +1 (Action increases, system degrades)
        return {"dU": -1, "dE": -1, "dC": 1, "j": 1}
    
    elif rule_type == "repair":
        # Edge Creation: U increases (+1), E increases (+1), C decreases (-1)
        # j = 1 - 1 + (-1) = -1 (Action decreases, system adapts)
        return {"dU": 1, "dE": 1, "dC": -1, "j": -1}
    
    return {"j": 0}

def generate_topology(species_type, steps=8):
    """
    Generates the Multiway Causal Graph.
    Nodes = FrozenSet of edges (configurations).
    """
    # Initial state: Ring graph of 3 nodes
    initial_state = frozenset({(1,2), (2,3), (3,1)})
    
    MWG = nx.DiGraph()
    MWG.add_node(initial_state, layer=0)
    
    frontier = [initial_state]
    seen = {initial_state}
    
    for step in range(1, steps + 1):
        new_frontier = []
        for state in frontier:
            # Dead state (empty) has no futures
            if not state: 
                continue
                
            # --- APPLY RULES ---
            
            # 1. ENTROPY RULE (Universal)
            # Try removing each edge
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

            # 2. REPAIR RULE (Plant Only)
            # Try closing triangles (transitivity)
            if species_type == "plant":
                # Rebuild graph to find open triangles
                G_temp = nx.Graph(list(state))
                nodes = list(G_temp.nodes())
                existing = set(state)
                
                possible_repairs = []
                for n in nodes:
                    neighbors = list(G_temp.neighbors(n))
                    for n1, n2 in combinations(neighbors, 2):
                        if not G_temp.has_edge(n1, n2):
                            # Found open triangle n1-n-n2
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
# 2. TOPOLOGICAL RISK METRICS
# ==========================================

def analyze_pathway_topology(MWG, beta=1.0):
    """
    Computes N_eff (Diversity), Fisher Info (Sensitivity), and Spectral Gap.
    """
    # Find root node by layer (initial state at layer 0)
    root = [n for n, attr in MWG.nodes(data=True) if attr.get('layer', -1) == 0][0]

    # Find terminal nodes (can be leaves or nodes at max layer)
    leaves = [n for n, d in MWG.out_degree() if d == 0]
    if not leaves:
        # If no leaves (cyclic graph), use nodes at maximum layer
        max_layer = max(attr.get('layer', 0) for _, attr in MWG.nodes(data=True))
        leaves = [n for n, attr in MWG.nodes(data=True) if attr.get('layer', -1) == max_layer]
    
    # 1. Enumerate all trajectories (paths from root to leaves/end of simulation)
    # Note: In large graphs, we would use Monte Carlo. Here we enumerate for exactness.
    paths = []
    actions = []

    # Limit path enumeration to prevent explosion in cyclic graphs
    max_paths_per_leaf = 1000

    for leaf in leaves:
        try:
            # Use cutoff to limit path length (prevent infinite loops in cycles)
            path_generator = nx.all_simple_paths(MWG, root, leaf, cutoff=20)
            leaf_paths = []
            for path in path_generator:
                # Calculate Action J[gamma] for this path
                J_gamma = 0
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    J_gamma += MWG[u][v]['cost']
                paths.append(path)
                actions.append(J_gamma)
                leaf_paths.append(path)

                # Limit paths per leaf to prevent explosion
                if len(leaf_paths) >= max_paths_per_leaf:
                    break
        except nx.NetworkXNoPath:
            continue

    if not actions:
        return 0, 0, 0 # Dead system or no paths found
        
    actions = np.array(actions)
    
    # 2. Compute Probability Distribution P(gamma) ~ exp(-beta * J)
    # Using softmax for numerical stability
    logits = -beta * actions
    logits -= np.max(logits) # Shift for stability
    unnorm_probs = np.exp(logits)
    partition_Z = np.sum(unnorm_probs)
    probs = unnorm_probs / partition_Z
    
    # --- METRIC 1: Effective Path Diversity (N_eff) ---
    # Shannon Entropy H = -Sum p ln p
    # N_eff = exp(H)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    n_eff = np.exp(entropy)
    
    # --- METRIC 2: Topological Sensitivity (Fisher Info) ---
    # In statistical mechanics, Fisher Info w.r.t beta is the Variance of the Energy (Action)
    # I(beta) = Var(J) = E[J^2] - E[J]^2
    expected_J = np.sum(probs * actions)
    expected_sq_J = np.sum(probs * (actions ** 2))
    fisher_info = expected_sq_J - (expected_J ** 2)
    
    # --- METRIC 3: Structural Connectivity (Spectral Gap) ---
    # We treat the Multiway Graph as an undirected structure to test connectedness
    # lambda_2 of the Normalized Laplacian
    undir_G = MWG.to_undirected()
    # Use largest connected component to avoid 0 if already fragmented (we want to measure the 'main' structure)
    largest_cc = max(nx.connected_components(undir_G), key=len)
    subgraph = undir_G.subgraph(largest_cc)
    
    if len(subgraph) > 1:
        L = nx.normalized_laplacian_matrix(subgraph)
        eigenvalues = scipy.linalg.eigvalsh(L.todense())
        eigenvalues.sort()
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
    else:
        spectral_gap = 0
        
    return n_eff, fisher_info, spectral_gap

# ==========================================
# 3. COMPARATIVE ANALYSIS & PLOTTING
# ==========================================

def run_analysis():
    print("Generating Multiway Causal Graphs...")
    # Generate systems
    mwg_glass = generate_topology("glass", steps=6)
    mwg_plant = generate_topology("plant", steps=6)
    
    print(f"Glass Nodes: {mwg_glass.number_of_nodes()}")
    print(f"Plant Nodes: {mwg_plant.number_of_nodes()}")

    # Analyze over a range of optimization pressures (beta)
    # Low beta = Exploration / High Uncertainty
    # High beta = Strong Optimization / Resource Constraint
    betas = np.linspace(0.1, 5.0, 20)
    
    glass_neff, glass_fisher = [], []
    plant_neff, plant_fisher = [], []
    
    # Spectral gap is topological, invariant to beta (calculated once)
    _, _, glass_spec = analyze_pathway_topology(mwg_glass)
    _, _, plant_spec = analyze_pathway_topology(mwg_plant)
    
    print("\nComputing Risk Metrics across optimization regimes...")
    for b in betas:
        n, f, _ = analyze_pathway_topology(mwg_glass, beta=b)
        glass_neff.append(n)
        glass_fisher.append(f)
        
        n, f, _ = analyze_pathway_topology(mwg_plant, beta=b)
        plant_neff.append(n)
        plant_fisher.append(f)

    # --- CREATE COMPARATIVE TABLE ---
    # Use beta = 1.0 for the table (balanced optimization regime)
    idx_mid = len(betas) // 2
    beta_ref = betas[idx_mid]

    # Get metrics at reference beta
    glass_neff_ref = glass_neff[idx_mid]
    plant_neff_ref = plant_neff[idx_mid]
    glass_fisher_ref = glass_fisher[idx_mid]
    plant_fisher_ref = plant_fisher[idx_mid]

    # Calculate average metrics across all beta values
    glass_neff_avg = np.mean(glass_neff)
    plant_neff_avg = np.mean(plant_neff)
    glass_fisher_avg = np.mean(glass_fisher)
    plant_fisher_avg = np.mean(plant_fisher)

    # Create comparison table
    fig_table = plt.figure(figsize=(12, 8))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('tight')
    ax_table.axis('off')

    table_data = [
        ["Metric", "Glass", "Plant"],
        ["Total States (Multiway Graph)",
         f"{mwg_glass.number_of_nodes()}",
         f"{mwg_plant.number_of_nodes()}"],
        ["$N_{{eff}}$ (at β={:.1f})".format(beta_ref),
         f"{glass_neff_ref:.2f}",
         f"{plant_neff_ref:.2f}"],
        ["Mean $N_{{eff}}$ (across all β)",
         f"{glass_neff_avg:.2f}",
         f"{plant_neff_avg:.2f}"],
        ["Fisher Info (at β={:.1f})".format(beta_ref),
         f"{glass_fisher_ref:.3f}",
         f"{plant_fisher_ref:.3f}"],
        ["Mean Fisher Info (across all β)",
         f"{glass_fisher_avg:.3f}",
         f"{plant_fisher_avg:.3f}"],
        ["Spectral Gap $\\lambda_2$",
         f"{glass_spec:.4f}",
         f"{plant_spec:.4f}"],
    ]

    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.50, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Style Glass column (alternating rows for readability)
    for i in range(1, len(table_data)):
        if i % 2 == 1:
            table[(i, 1)].set_facecolor('#f5f5f5')  # Light gray
            table[(i, 2)].set_facecolor('#f5f5f5')  # Light gray

    plt.savefig("topological_risk_table.png", dpi=300, bbox_inches='tight')
    print("Table saved to topological_risk_table.png")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Effective Path Diversity (N_eff)
    ax = axes[0]
    ax.plot(betas, glass_neff, 'o-', color='salmon', label='Glass (Degradation)', linewidth=2)
    ax.plot(betas, plant_neff, 's-', color='teal', label='Plant (Adaptive)', linewidth=2)
    ax.set_title("Effective Path Diversity ($N_{eff}$)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Optimization Pressure ($\\beta$)")
    ax.set_ylabel("Effective # of Futures")
    ax.legend()

    # 2. Topological Sensitivity (Fisher Info)
    ax = axes[1]
    ax.plot(betas, glass_fisher, 'o--', color='salmon', label='Glass', linewidth=2)
    ax.plot(betas, plant_fisher, 's-', color='teal', label='Plant', linewidth=2)
    ax.set_title("Topological Sensitivity (Fisher Info)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Optimization Pressure ($\\beta$)")
    ax.set_ylabel("Sensitivity $I(\\beta) = Var(J)$")
    ax.fill_between(betas, glass_fisher, alpha=0.1, color='salmon')
    ax.text(betas[len(betas)//2], max(glass_fisher)/2, "High Sensitivity\n(Tipping Risk)",
            color='red', ha='center')

    # 3. Spectral Gap (Connectivity)
    ax = axes[2]
    categories = ['Glass', 'Plant']
    values = [glass_spec, plant_spec]
    bars = ax.bar(categories, values, color=['salmon', 'teal'], alpha=0.8, width=0.5)
    ax.set_title("Structural Connectivity (Spectral Gap $\\lambda_2$)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Algebraic Connectivity")
    ax.set_ylim(0, max(values)*1.2)

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("topological_risk_metrics.png", dpi=300, bbox_inches='tight')
    print("Plot saved to topological_risk_metrics.png")
    plt.show()

if __name__ == "__main__":
    run_analysis()