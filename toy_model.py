import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import logsumexp
from scipy.optimize import minimize
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

class SurvivalLandscape:
    """
    Cellular automaton representing a survival landscape where agents
    must navigate from start to goal, optimizing resources vs. costs.
    """
    def __init__(self, size=25, n_hazards=20, n_resources=15, seed=42):
        np.random.seed(seed)
        self.size = size
        self.grid = np.zeros((size, size))
        
        # Define start and goal
        self.start = (2, 2)
        self.goal = (size-3, size-3)
        
        # Create multiple "corridors" with different risk-reward profiles
        # Corridor 1: Resource-rich but hazardous (upper path)
        for i in range(5, 15):
            self.grid[5, i] = np.random.uniform(2, 5)  # Resources
            if np.random.rand() > 0.6:
                self.grid[6, i] = -np.random.uniform(4, 7)  # Hazards
        
        # Corridor 2: Safe but resource-poor (middle path)
        for i in range(8, 18):
            if np.random.rand() > 0.7:
                self.grid[12, i] = np.random.uniform(0.5, 2)  # Few resources
        
        # Corridor 3: High-risk high-reward (lower path)
        for i in range(10, 20):
            if np.random.rand() > 0.5:
                self.grid[18, i] = np.random.uniform(3, 6)  # High resources
            if np.random.rand() > 0.5:
                self.grid[19, i] = -np.random.uniform(5, 9)  # High hazards
        
        # Add random scattered resources and hazards
        for _ in range(n_hazards):
            x, y = np.random.randint(3, size-3, 2)
            if (x, y) != self.start and (x, y) != self.goal:
                if self.grid[x, y] == 0:  # Don't override corridors
                    self.grid[x, y] = -np.random.uniform(3, 8)
        
        for _ in range(n_resources):
            x, y = np.random.randint(3, size-3, 2)
            if (x, y) != self.start and (x, y) != self.goal:
                if self.grid[x, y] == 0:  # Don't override
                    self.grid[x, y] = np.random.uniform(1, 4)
    
    def compute_action_score(self, trajectory, utility_weight=1.0, effort_weight=1.0):
        """
        Compute J[γ] = Effort - Utility + Constraints
        utility_weight and effort_weight allow heterogeneous preferences
        """
        effort = len(trajectory) * 0.5 * effort_weight
        utility = 0
        constraints = 0
        
        for (x, y) in trajectory:
            # Collect resources (negative in J means good)
            utility += max(0, self.grid[x, y]) * utility_weight
            # Pay penalty for hazards
            constraints += max(0, -self.grid[x, y])
            
        # Penalty for not reaching goal
        if trajectory[-1] != self.goal:
            constraints += 50
            
        return effort - utility + constraints
    
    def generate_trajectory(self, utility_weight=1.0, risk_tolerance=1.0, seed=None):
        """
        Generate a single trajectory for an agent with specific preferences.
        
        Parameters:
        - utility_weight: how much the agent values resources
        - risk_tolerance: higher = more willing to explore risky paths
        """
        if seed is not None:
            np.random.seed(seed)
            
        path = [self.start]
        current = self.start
        max_steps = self.size * 3
        visited = set([self.start])
        
        for step in range(max_steps):
            if current == self.goal:
                break
            
            # Possible moves (4-connectivity)
            moves = []
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                next_pos = (current[0]+dx, current[1]+dy)
                if (0 <= next_pos[0] < self.size and 
                    0 <= next_pos[1] < self.size):
                    moves.append(next_pos)
            
            if not moves:
                break
            
            # Score each move based on agent's preferences
            scores = []
            for move in moves:
                # Distance to goal (Manhattan)
                goal_dist = abs(move[0] - self.goal[0]) + abs(move[1] - self.goal[1])
                score = -goal_dist * 0.5
                
                # Resource/hazard value weighted by preferences
                cell_value = self.grid[move[0], move[1]]
                if cell_value > 0:  # Resource
                    score += cell_value * utility_weight * 2
                else:  # Hazard
                    score += cell_value * 2  # Negative score for hazards
                
                # Penalty for revisiting
                if move in visited:
                    score -= 5
                
                scores.append(score)
            
            # Softmax selection with risk tolerance
            scores = np.array(scores)
            probs = np.exp(scores * risk_tolerance) / np.sum(np.exp(scores * risk_tolerance))
            
            # Choose next move
            idx = np.random.choice(len(moves), p=probs)
            current = moves[idx]
            path.append(current)
            visited.add(current)
        
        return path

def generate_heterogeneous_storylines(landscape, n_agents=500, seed=42):
    """
    Generate diverse storylines from agents with heterogeneous preferences.
    This represents different optimization processes in the real world.
    """
    np.random.seed(seed)
    
    trajectories = []
    agent_params = []
    
    for i in range(n_agents):
        # Sample heterogeneous preferences from realistic distributions
        # Some agents value resources highly, others less so
        utility_weight = np.random.lognormal(0, 0.5)  # Mean ~1, varies
        
        # Some agents are risk-seeking, others risk-averse
        risk_tolerance = np.random.gamma(2, 0.5)  # Mean ~1, varies
        
        # Generate trajectory based on these preferences
        trajectory = landscape.generate_trajectory(
            utility_weight=utility_weight,
            risk_tolerance=risk_tolerance,
            seed=seed + i
        )
        
        trajectories.append(trajectory)
        agent_params.append({
            'utility_weight': utility_weight,
            'risk_tolerance': risk_tolerance
        })
    
    return trajectories, agent_params

def compute_action_scores(landscape, trajectories, agent_params):
    """
    Compute action scores for all trajectories, accounting for 
    heterogeneous utility functions.
    """
    action_scores = []
    
    for traj, params in zip(trajectories, agent_params):
        J = landscape.compute_action_score(
            traj, 
            utility_weight=params['utility_weight'],
            effort_weight=1.0
        )
        action_scores.append(J)
    
    return np.array(action_scores)

def estimate_beta_from_observations(action_scores, method='moment_matching'):
    """
    Infer β from observed distribution of action scores.
    
    After all realizations, we estimate what optimization strength
    best explains the observed distribution.
    
    Methods:
    - 'moment_matching': Use mean and variance relationship
    - 'variance': Use the relationship between variance and β
    - 'mle': Maximum likelihood estimation
    """
    action_scores = np.array(action_scores)
    
    if method == 'moment_matching':
        # For the exponential distribution P(γ) ∝ exp(-β J[γ]):
        # When β is small (antifragile): wide distribution, high variance
        # When β is large (fragile): narrow distribution, low variance
        # Use: β ∝ 1/std(J) to capture concentration
        std_J = np.std(action_scores)
        if std_J > 0:
            # Scale to give reasonable β values (0.1 to 10)
            beta_est = 1.0 / std_J
        else:
            beta_est = 1.0
    
    elif method == 'variance':
        # Simple variance-based approach
        var_J = np.var(action_scores)
        if var_J > 0:
            beta_est = 1.0 / var_J
        else:
            beta_est = 1.0
            
    elif method == 'mle':
        # Maximum likelihood: find β that best fits observed distribution
        def neg_log_likelihood(beta):
            if beta <= 0:
                return np.inf
            # Uniform P0
            log_Z = logsumexp(-beta * action_scores)
            # Log-likelihood of observed samples
            log_probs = -beta * action_scores - log_Z
            return -np.sum(log_probs)
        
        result = minimize(neg_log_likelihood, x0=[1.0], bounds=[(0.01, 100)])
        beta_est = result.x[0]
    
    return beta_est

def compute_risk_metrics(action_scores, beta, P0=None):
    """
    Compute all risk metrics: Z, H, F, N_eff
    """
    action_scores = np.array(action_scores)
    
    if P0 is None:
        P0 = np.ones(len(action_scores)) / len(action_scores)
    else:
        P0 = np.array(P0)
        P0 = P0 / np.sum(P0)
    
    # Compute partition function Z using log-sum-exp for numerical stability
    log_Z = logsumexp(-beta * action_scores + np.log(P0))
    Z = np.exp(log_Z)
    
    # Compute probabilities P(γ)
    log_P = -beta * action_scores + np.log(P0) - log_Z
    P = np.exp(log_P)
    
    # Expected action score
    J_avg = np.sum(P * action_scores)
    
    # Entropy H = -sum P log P (handle P=0 gracefully)
    H = -np.sum(P[P > 1e-15] * np.log(P[P > 1e-15]))
    
    # Effective number of paths
    N_eff = np.exp(H)
    
    # Risk potential F = -β^(-1) ln Z
    F = -log_Z / beta if beta > 0 else np.inf
    
    # Verify entropy relation: H = ln Z + β⟨J⟩
    H_check = log_Z + beta * J_avg
    
    # Fisher information (variance of action scores)
    var_J = np.sum(P * (action_scores - J_avg)**2)
    
    return {
        'Z': Z,
        'H': H,
        'N_eff': N_eff,
        'F': F,
        'J_avg': J_avg,
        'P': P,
        'H_check': H_check,
        'H_error': abs(H - H_check),
        'fisher_info': var_J
    }

# ============================================================================
# GENERATE THREE SCENARIOS WITH DIFFERENT PREFERENCE STRUCTURES
# ============================================================================

print("="*85)
print("GENERATING THREE SCENARIOS: DIVERSE vs ALIGNED PREFERENCES")
print("="*85)

landscape = SurvivalLandscape(size=25, n_hazards=20, n_resources=15, seed=42)

# Scenario 1: DIVERSE PREFERENCES (agents optimize different objectives - LOW β expected)
print("\n[1/3] Generating agents with DIVERSE PREFERENCES...")
print("     → Agents value resources differently and have varied risk attitudes")
np.random.seed(42)
trajectories_diverse = []
params_diverse = []
for i in range(100000):
    # High variance in preferences: agents want different things
    utility_weight = np.random.lognormal(0, 0.8)  # Some value resources highly, others don't
    risk_tolerance = np.random.uniform(0.5, 2.5)  # Some risk-averse, others risk-seeking
    traj = landscape.generate_trajectory(utility_weight, risk_tolerance, seed=42+i)
    trajectories_diverse.append(traj)
    params_diverse.append({'utility_weight': utility_weight, 'risk_tolerance': risk_tolerance})

action_scores_diverse = compute_action_scores(landscape, trajectories_diverse, params_diverse)
beta_diverse = estimate_beta_from_observations(action_scores_diverse, method='moment_matching')
metrics_diverse = compute_risk_metrics(action_scores_diverse, beta_diverse)
print(f"   → Std(J) = {np.std(action_scores_diverse):.4f} (wide spread in outcomes)")
print(f"   → Estimated β = {beta_diverse:.4f}")
print(f"   → N_eff = {metrics_diverse['N_eff']:.2f}")

# Scenario 2: MODERATELY ALIGNED PREFERENCES (some agreement - MEDIUM β expected)  
print("\n[2/3] Generating agents with MODERATELY ALIGNED PREFERENCES...")
print("     → Agents have somewhat similar goals but retain some individuality")
np.random.seed(100)
trajectories_moderate = []
params_moderate = []
for i in range(500):
    # Moderate variance: agents somewhat agree on what's valuable
    utility_weight = np.random.lognormal(0, 0.4)
    risk_tolerance = np.random.uniform(0.9, 1.6)
    traj = landscape.generate_trajectory(utility_weight, risk_tolerance, seed=100+i)
    trajectories_moderate.append(traj)
    params_moderate.append({'utility_weight': utility_weight, 'risk_tolerance': risk_tolerance})

action_scores_moderate = compute_action_scores(landscape, trajectories_moderate, params_moderate)
beta_moderate = estimate_beta_from_observations(action_scores_moderate, method='moment_matching')
metrics_moderate = compute_risk_metrics(action_scores_moderate, beta_moderate)
print(f"   → Std(J) = {np.std(action_scores_moderate):.4f} (moderate spread)")
print(f"   → Estimated β = {beta_moderate:.4f}")
print(f"   → N_eff = {metrics_moderate['N_eff']:.2f}")

# Scenario 3: ALIGNED PREFERENCES (agents optimize similarly - HIGH β expected)
print("\n[3/3] Generating agents with ALIGNED PREFERENCES...")
print("     → Agents share similar goals and strategies")
np.random.seed(200)
trajectories_aligned = []
params_aligned = []
for i in range(500):
    # Low variance: agents all want similar things and optimize similarly
    utility_weight = np.random.lognormal(0, 0.15)  # Everyone values resources similarly
    risk_tolerance = np.random.uniform(1.1, 1.4)   # Similar risk attitudes
    traj = landscape.generate_trajectory(utility_weight, risk_tolerance, seed=200+i)
    trajectories_aligned.append(traj)
    params_aligned.append({'utility_weight': utility_weight, 'risk_tolerance': risk_tolerance})

action_scores_aligned = compute_action_scores(landscape, trajectories_aligned, params_aligned)
beta_aligned = estimate_beta_from_observations(action_scores_aligned, method='moment_matching')
metrics_aligned = compute_risk_metrics(action_scores_aligned, beta_aligned)
print(f"   → Std(J) = {np.std(action_scores_aligned):.4f} (narrow spread - convergence)")
print(f"   → Estimated β = {beta_aligned:.4f}")
print(f"   → N_eff = {metrics_aligned['N_eff']:.2f}")

print("\nDone! Generating figures...\n")

# ============================================================================
# FIGURE 1: Landscapes and Realized Trajectories
# ============================================================================

fig1, axes = plt.subplots(1, 4, figsize=(16, 4))

# Panel A: Landscape only
ax = axes[0]
im = ax.imshow(landscape.grid, cmap='RdYlGn', origin='lower', vmin=-8, vmax=4)
ax.plot(landscape.start[1], landscape.start[0], 'o', color='darkblue', 
        markersize=12, label='Start', markeredgecolor='white', markeredgewidth=1.5)
ax.plot(landscape.goal[1], landscape.goal[0], '*', color='gold', 
        markersize=18, label='Goal', markeredgecolor='black', markeredgewidth=1)
ax.set_title('(A) Survival Landscape', fontweight='bold')
ax.set_xlabel('Y coordinate')
ax.set_ylabel('X coordinate')
ax.legend(loc='upper right', framealpha=0.9)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Value (Green=Resource, Red=Hazard)', fontsize=9)

# Panels B, C, D: Realized trajectories for each scenario
scenarios = [
    (trajectories_diverse, metrics_diverse, '(B) Diverse Preferences', 'dodgerblue', beta_diverse),
    (trajectories_moderate, metrics_moderate, '(C) Moderately Aligned', 'darkorange', beta_moderate),
    (trajectories_aligned, metrics_aligned, '(D) Aligned Preferences', 'crimson', beta_aligned)
]

for idx, (trajs, metrics, title, color, beta_val) in enumerate(scenarios):
    ax = axes[idx + 1]
    ax.imshow(landscape.grid, cmap='RdYlGn', origin='lower', alpha=0.25, vmin=-8, vmax=4)
    
    # Plot sample of trajectories
    sample_indices = np.random.choice(len(trajs), min(50, len(trajs)), replace=False)
    for i in sample_indices:
        traj = trajs[i]
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(ys, xs, alpha=0.4, color=color, linewidth=1.5)
    
    ax.plot(landscape.start[1], landscape.start[0], 'o', color='darkblue', 
            markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(landscape.goal[1], landscape.goal[0], '*', color='gold', 
            markersize=16, markeredgecolor='black', markeredgewidth=1)
    ax.set_title(f'{title}\nβ={beta_val:.3f}, N_eff={metrics["N_eff"]:.1f}', fontweight='bold')
    ax.set_xlabel('Y coordinate')
    
    if idx == 0:
        ax.set_ylabel('X coordinate')

plt.tight_layout()
plt.savefig('figure1_realized_trajectories.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure1_realized_trajectories.png")
plt.show()

# ============================================================================
# FIGURE 2: Comparative Risk Metrics
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

scenario_names = ['Diverse\nPreferences', 'Moderately\nAligned', 'Aligned\nPreferences']
betas_all = [beta_diverse, beta_moderate, beta_aligned]
colors = ['dodgerblue', 'darkorange', 'crimson']

metrics_all = [metrics_diverse, metrics_moderate, metrics_aligned]

# Panel A: Partition Function Z
ax = axes[0, 0]
Zs = [m['Z'] for m in metrics_all]
bars = ax.bar(scenario_names, Zs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Partition Function Z', fontweight='bold')
ax.set_title('(A) Adaptive Capacity', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
for i, (bar, val) in enumerate(zip(bars, Zs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Panel B: Effective Number of Paths
ax = axes[0, 1]
N_effs = [m['N_eff'] for m in metrics_all]
bars = ax.bar(scenario_names, N_effs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('N_eff = exp(H)', fontweight='bold')
ax.set_title('(B) Trajectory Diversity', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
for i, (bar, val) in enumerate(zip(bars, N_effs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Panel C: Risk Potential F
ax = axes[1, 0]
Fs = [m['F'] for m in metrics_all]
bars = ax.bar(scenario_names, Fs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Risk Potential F', fontweight='bold')
ax.set_title('(C) Vulnerability Metric', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
for i, (bar, val) in enumerate(zip(bars, Fs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Panel D: Estimated β
ax = axes[1, 1]
bars = ax.bar(scenario_names, betas_all, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Estimated β', fontweight='bold')
ax.set_title('(D) Inferred Optimization Strength', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.axhline(y=beta_diverse, color='blue', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=beta_homogeneous, color='red', linestyle='--', alpha=0.3, linewidth=1)
for i, (bar, val) in enumerate(zip(bars, betas_all)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_comparative_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure2_comparative_metrics.png")
plt.show()

# ============================================================================
# FIGURE 3: Action Score Distributions and β Estimation
# ============================================================================

fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Distribution of action scores for all scenarios
ax = axes[0, 0]
for scores, label, color in [
    (action_scores_diverse, 'Diverse Preferences', 'dodgerblue'),
    (action_scores_moderate, 'Moderately Aligned', 'darkorange'),
    (action_scores_aligned, 'Aligned Preferences', 'crimson')
]:
    ax.hist(scores, bins=40, alpha=0.5, label=label, color=color, edgecolor='black', linewidth=0.8)

ax.set_xlabel('Action Score J[γ]', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('(A) Distribution of Action Scores', fontweight='bold', fontsize=13)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Panel B: Variance vs β relationship
ax = axes[0, 1]
variances = [np.var(action_scores_diverse), np.var(action_scores_moderate), np.var(action_scores_homogeneous)]
ax.scatter(betas_all, variances, s=200, c=colors, edgecolor='black', linewidth=2, zorder=5)
for i, name in enumerate(scenario_names):
    ax.annotate(name, (betas_all[i], variances[i]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')

# Add theoretical curve: Var ≈ 1/β
beta_theory = np.linspace(0.01, max(betas_all)*1.2, 100)
var_theory = 1.0 / beta_theory
ax.plot(beta_theory, var_theory, 'k--', alpha=0.5, linewidth=2, label='Theory: Var ≈ 1/β')

ax.set_xlabel('Estimated β', fontweight='bold')
ax.set_ylabel('Variance of J[γ]', fontweight='bold')
ax.set_title('(B) β vs Variance Relationship', fontweight='bold', fontsize=13)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

# Panel C: Entropy validation H = ln Z + β⟨J⟩
ax = axes[1, 0]
H_direct = [m['H'] for m in metrics_all]
H_theory = [m['H_check'] for m in metrics_all]
x_pos = np.arange(len(scenario_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, H_direct, width, label='H (direct)', 
               color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, H_theory, width, label='ln Z + β⟨J⟩', 
               color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Entropy', fontweight='bold')
ax.set_title('(C) Theoretical Validation', fontweight='bold', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels(scenario_names)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Panel D: Probability distributions
ax = axes[1, 1]
for scores, metrics, label, color in [
    (action_scores_diverse, metrics_diverse, 'Diverse', 'dodgerblue'),
    (action_scores_moderate, metrics_moderate, 'Moderately Aligned', 'darkorange'),
    (action_scores_aligned, metrics_aligned, 'Aligned', 'crimson')
]:
    P = metrics['P']
    sorted_indices = np.argsort(scores)
    sorted_P = P[sorted_indices]
    
    ax.plot(range(len(sorted_P)), sorted_P, linewidth=2.5, 
            label=f'{label} (N_eff={metrics["N_eff"]:.1f})', 
            color=color, alpha=0.85)

ax.set_xlabel('Trajectory Index (sorted by J[γ])', fontweight='bold')
ax.set_ylabel('Probability P(γ)', fontweight='bold')
ax.set_title('(D) Optimized Probability Distributions', fontweight='bold', fontsize=13)
ax.set_yscale('log')
ax.legend(fontsize=10, framealpha=0.95, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure3_estimation_validation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure3_estimation_validation.png")
plt.show()

# ============================================================================
# FIGURE 4: Agent Preference Distributions
# ============================================================================

fig4, axes = plt.subplots(2, 3, figsize=(15, 10))

scenarios_params = [
    (params_diverse, 'Diverse Preferences', 'dodgerblue'),
    (params_moderate, 'Moderately Aligned', 'darkorange'),
    (params_aligned, 'Aligned Preferences', 'crimson')
]

for idx, (params, title, color) in enumerate(scenarios_params):
    # Utility weight distribution
    ax = axes[0, idx]
    utility_weights = [p['utility_weight'] for p in params]
    ax.hist(utility_weights, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Utility Weight', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{title}\nUtility Preferences', fontweight='bold')
    ax.axvline(np.mean(utility_weights), color='red', linestyle='--', linewidth=2, 
               label=f'Mean={np.mean(utility_weights):.2f}')
    ax.axvline(np.median(utility_weights), color='orange', linestyle='-.', linewidth=2, 
               label=f'Median={np.median(utility_weights):.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Risk tolerance distribution
    ax = axes[1, idx]
    risk_tolerances = [p['risk_tolerance'] for p in params]
    ax.hist(risk_tolerances, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Risk Tolerance', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Risk Tolerance Distribution', fontweight='bold')
    ax.axvline(np.mean(risk_tolerances), color='red', linestyle='--', linewidth=2, 
               label=f'Mean={np.mean(risk_tolerances):.2f}')
    ax.axvline(np.median(risk_tolerances), color='orange', linestyle='-.', linewidth=2, 
               label=f'Median={np.median(risk_tolerances):.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('figure4_agent_preferences.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure4_agent_preferences.png")
plt.show()

# ============================================================================
# NUMERICAL RESULTS TABLE
# ============================================================================

print("\n" + "="*85)
print(" NUMERICAL RESULTS: β ESTIMATED FROM REALIZED STORYLINES")
print("="*85)

for name, metrics, beta_val, color in [
    ('DIVERSE PREFERENCES', metrics_diverse, beta_diverse, 'dodgerblue'),
    ('MODERATELY ALIGNED PREFERENCES', metrics_moderate, beta_moderate, 'darkorange'),
    ('ALIGNED PREFERENCES', metrics_aligned, beta_aligned, 'crimson')
]:
    print(f"\n {name}")
    print("-" * 85)
    print(f"  Estimated β (from data):       {beta_val:12.6f}")
    print(f"  Partition Function Z:          {metrics['Z']:12.6f}")
    print(f"  Entropy H:                     {metrics['H']:12.6f}")
    print(f"  Effective Paths N_eff:         {metrics['N_eff']:12.2f}")
    print(f"  Risk Potential F:              {metrics['F']:12.6f}")
    print(f"  Expected Action ⟨J⟩:           {metrics['J_avg']:12.6f}")
    print(f"  Fisher Information (Var[J]):   {metrics['fisher_info']:12.6f}")
    print(f"  Entropy check (ln Z + β⟨J⟩):   {metrics['H_check']:12.6f}")
    print(f"  Verification error:            {metrics['H_error']:12.2e}")

print("\n" + "="*85)