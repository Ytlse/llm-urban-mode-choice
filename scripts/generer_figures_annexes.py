#!/usr/bin/env python3
"""Generation of all high-definition figures for the appendix volume.

Produces publication-quality charts (300 DPI) in
article-court/appendices/images/ of the papers repository (`PAPER_DIR`, ticket 115), or in
`outputs/figures/` when that repository is absent (the public copy).

The nine figures draw values written in this file: the script reads no data, and regenerating
a figure reproduces its drawing, not a measurement.
"""

import math
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier  # noqa: E402

# Output directory
OUTPUT_DIR = sortie_papier("article-court", "appendices", "images")
exiger_depot_papiers(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global plot configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

PALETTE = {
    'primary': '#1f77b4',     # Classic blue
    'secondary': '#ff7f0e',   # Orange
    'success': '#2ca02c',     # Green
    'danger': '#d62728',      # Red
    'purple': '#9467bd',      # Purple
    'brown': '#8c564b',       # Brown
    'pink': '#e377c2',        # Pink
    'gray': '#7f7f7f',        # Grey
    'dark': '#1a252f',        # Midnight blue
    'light': '#f8f9fa',       # Off-white
}

print(f"Generating the figures in {OUTPUT_DIR}...")

# ==============================================================================
# 1. FIG: EBBINGHAUS DECAY & REINFORCEMENT (Chapter 11 - Memory)
# ==============================================================================
def generer_ebbinghaus_decay():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    t = np.linspace(0, 15, 500)
    
    # Park et al. (2023): decay 0.995 / hour -> lambda = 0.12 / day -> half-life 5.78 days
    lambda_park = 24 * math.log(1 / 0.995)
    r_park = np.exp(-lambda_park * t)
    
    # Our mobility calibration: constant 2.8 days -> lambda = 1/2.8 = 0.357 / day -> half-life 1.94 day
    lambda_transport = 1.0 / 2.8
    r_transport = np.exp(-lambda_transport * t)
    
    # Reinforced memory (recall on day 3)
    r_reinforced = np.zeros_like(t)
    mask1 = t <= 3.0
    mask2 = t > 3.0
    r_reinforced[mask1] = np.exp(-lambda_transport * t[mask1])
    # On day 3, recall: activation jump and lifespan extension
    val_at_3 = np.exp(-lambda_transport * 3.0)
    boosted_val = min(1.0, val_at_3 + 0.45)
    # Slower decay after reinforcement (doubled half-life)
    r_reinforced[mask2] = boosted_val * np.exp(- (lambda_transport / 1.8) * (t[mask2] - 3.0))
    
    ax.plot(t, r_park, label='Park et al. (2023) Generative Agents (Half-life = 5.8 d)', 
            color=PALETTE['primary'], linestyle='--', linewidth=2)
    ax.plot(t, r_transport, label='Our Calibrated Mobility Decay (Half-life = 1.94 d)', 
            color=PALETTE['danger'], linewidth=2.5)
    ax.plot(t, r_reinforced, label='Memory with Recall Reinforcement at Day 3', 
            color=PALETTE['success'], linewidth=2)
    
    # Minimum recall threshold
    ax.axhline(y=0.15, color=PALETTE['gray'], linestyle=':', label='Recall Retrieval Threshold (0.15)')
    
    # Annotations
    ax.scatter([1.94], [0.5], color=PALETTE['danger'], s=50, zorder=5)
    ax.annotate(r'$\tau_{1/2} = 1.94$ d', xy=(1.94, 0.5), xytext=(2.6, 0.55),
                arrowprops=dict(arrowstyle='->', color=PALETTE['danger'], lw=1.2),
                fontweight='bold', color=PALETTE['danger'])
    
    ax.scatter([5.78], [0.5], color=PALETTE['primary'], s=50, zorder=5)
    ax.annotate(r'$\tau_{1/2} = 5.78$ d', xy=(5.78, 0.5), xytext=(6.5, 0.58),
                arrowprops=dict(arrowstyle='->', color=PALETTE['primary'], lw=1.2),
                fontweight='bold', color=PALETTE['primary'])
    
    ax.scatter([3.0], [boosted_val], color=PALETTE['success'], s=60, zorder=5)
    ax.annotate('Active Recall Event\n(+1 day lifespan boost)', xy=(3.0, boosted_val), xytext=(3.8, 0.82),
                arrowprops=dict(arrowstyle='->', color=PALETTE['success'], lw=1.2),
                fontweight='bold', color=PALETTE['success'])
    
    ax.set_title('Cognitive Memory Retention: Ebbinghaus Decay and Recall Reinforcement', pad=12)
    ax.set_xlabel('Elapsed Time since Event Inception (Days)')
    ax.set_ylabel(r'Memory Activation Strength $R(t) \in [0, 1]$')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    out_path = OUTPUT_DIR / "fig_ebbinghaus_decay.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 2. FIG: MEMORY RETRIEVAL RADAR CHART (Chapter 11 - Memory)
# ==============================================================================
def generer_retrieval_components():
    categories = ['Recency\n(Weight: 0.30)', 'Semantic Sim.\n(Weight: 0.20)', 
                  'Severity/Gravity\n(Weight: 0.20)', 'Recall Freq.\n(Weight: 0.20)', 
                  'Valence Impact\n(Weight: 0.10)']
    N = len(categories)
    
    # 3 event profiles
    vals_banal = [0.70, 0.40, 0.10, 0.85, 0.10]
    vals_accident = [0.35, 0.85, 0.90, 0.40, 0.95]
    vals_depeche = [0.95, 0.75, 0.70, 0.15, 0.60]
    
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    
    vals_banal += vals_banal[:1]
    vals_accident += vals_accident[:1]
    vals_depeche += vals_depeche[:1]
    
    fig, ax = plt.subplots(figsize=(7, 6.5), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='black', size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    ax.plot(angles, vals_banal, linewidth=2, linestyle='solid', color=PALETTE['gray'], label='Routine Commute (Score = 0.45)')
    ax.fill(angles, vals_banal, color=PALETTE['gray'], alpha=0.15)
    
    ax.plot(angles, vals_accident, linewidth=2.5, linestyle='solid', color=PALETTE['danger'], label='Severe Breakdown (Score = 0.63)')
    ax.fill(angles, vals_accident, color=PALETTE['danger'], alpha=0.2)
    
    ax.plot(angles, vals_depeche, linewidth=2.5, linestyle='solid', color=PALETTE['primary'], label='Morning Windstorm Shock (Score = 0.67)')
    ax.fill(angles, vals_depeche, color=PALETTE['primary'], alpha=0.2)
    
    ax.set_title('Five-Component Episodic Memory Retrieval Scoring Profile', size=12, weight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, facecolor='white')
    
    out_path = OUTPUT_DIR / "fig_retrieval_components.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 3. FIG: MEMORY COGNITIVE PIPELINE (Chapter 11 - Memory)
# ==============================================================================
def generer_memory_pipeline():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axis('off')
    
    boxes = [
        (0.04, 0.65, 0.27, 0.28, "1. Sensory Input & Events", "• Exogenous press dispatches\n• Lived travel delays & shocks\n• Morning severity rating (1-5)", PALETTE['primary']),
        (0.365, 0.65, 0.27, 0.28, "2. Short-Term Memory (STM)", "• Timestamped episode buffer\n• Trip observations\n• Pre-consolidation queue", '#3498db'),
        (0.69, 0.65, 0.27, 0.28, "3. Evening Consolidation", "• Triggered at 22:00 daily\n• Salient reflections & abstracts\n• Concept formation floor", PALETTE['purple']),
        
        (0.69, 0.18, 0.27, 0.28, "4. Long-Term Memory (LTM)", "• Episodic concepts repository\n• Dual extinction (wear vs cont.)\n• Ebbinghaus exponential decay", '#27ae60'),
        (0.365, 0.18, 0.27, 0.28, "5. Household 1-Hop Sharing", "• Evening dinner exchange\n• Shares concepts with family\n• Strict 1-hop viral barrier", PALETTE['secondary']),
        (0.04, 0.18, 0.27, 0.28, "6. Decision Context Retrieval", "• 5-component weighted score\n• Vehicle chain constraint filter\n• Jinja2 prompt injection", '#e67e22'),
    ]
    
    for x, y, w, h, title, subtitle, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                     linewidth=1.8, edgecolor=color, facecolor=color, alpha=0.08)
        ax.add_patch(rect)
        rect_border = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                            linewidth=1.8, edgecolor=color, facecolor='none')
        ax.add_patch(rect_border)
        
        header_bar = patches.FancyBboxPatch((x, y + h - 0.065), w, 0.065, boxstyle="round,pad=0.01,rounding_size=0.015",
                                           linewidth=0, facecolor=color, alpha=0.2)
        ax.add_patch(header_bar)
        
        ax.text(x + w/2, y + h - 0.033, title, horizontalalignment='center',
                verticalalignment='center', fontsize=9.5, fontweight='bold', color=color)
        ax.text(x + 0.02, y + h - 0.09, subtitle, horizontalalignment='left',
                verticalalignment='top', fontsize=8.5, color='#2c3e50', linespacing=1.4)
    
    arrow_props = dict(facecolor='#34495e', edgecolor='#34495e', width=2, headwidth=7, shrink=0.04)
    
    # 1 -> 2
    ax.annotate('', xy=(0.365, 0.79), xytext=(0.31, 0.79), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate('', xy=(0.69, 0.79), xytext=(0.635, 0.79), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate('', xy=(0.825, 0.46), xytext=(0.825, 0.65), arrowprops=arrow_props)
    # 4 -> 5
    ax.annotate('', xy=(0.635, 0.32), xytext=(0.69, 0.32), arrowprops=arrow_props)
    # 5 -> 6
    ax.annotate('', xy=(0.31, 0.32), xytext=(0.365, 0.32), arrowprops=arrow_props)
    
    # 6 -> Output
    ax.annotate('', xy=(0.175, 0.08), xytext=(0.175, 0.18), 
                arrowprops=dict(facecolor=PALETTE['danger'], edgecolor=PALETTE['danger'], width=2.2, headwidth=8))
    ax.text(0.175, 0.04, 'Injected into Decision Prompt (LLM / TypeSafe Classifier)', 
            fontsize=9.5, fontweight='bold', color=PALETTE['danger'], horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='#fdf2f2', ec=PALETTE['danger'], lw=1.2))

    ax.set_title('Complete Multi-Agent Cognitive Memory Architecture and Lifecycle', fontsize=12.5, weight='bold', pad=15)
    
    out_path = OUTPUT_DIR / "fig_memory_lifecycle.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 4. FIG: C4 CONTAINER TOPOLOGY (Chapter 15 - Architecture)
# ==============================================================================
def generer_c4_topology():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis('off')
    
    host_rect = patches.FancyBboxPatch((0.02, 0.03), 0.96, 0.92, boxstyle="round,pad=0.02,rounding_size=0.03",
                                      linewidth=1.5, edgecolor='#7f8c8d', facecolor='#f8f9fa')
    ax.add_patch(host_rect)
    ax.text(0.05, 0.92, "Docker Compose Host Environment (7 Containerized Microservices)", 
            fontsize=11, fontweight='bold', color='#34495e')
    
    containers = [
        (0.06, 0.58, 0.26, 0.26, "GAMA Simulation\nEngine", "Java / GAMA headless\nABM Physical Simulator", "Ports 3001, 6868", PALETTE['success']),
        (0.38, 0.58, 0.26, 0.26, "Simulation Controller\nOrchestrator", "Python 3.12 / FastAPI\nAgent State & Lifecycle", "Port 8002", PALETTE['primary']),
        (0.70, 0.58, 0.24, 0.26, "Redis Datastore\n& Token Bucket", "Redis 7 / In-memory\nDB0: State / DB1: Cache", "Port 6379", PALETTE['danger']),
        
        (0.06, 0.12, 0.26, 0.26, "Multi-Modal Routers\n(OTP & OSMnx)", "Java OTP 1.5 (x3)\nPython OSMnx Cache", "Ports 8080-8082, 8090", PALETTE['purple']),
        (0.38, 0.12, 0.26, 0.26, "SWRR LLM Gateway\n& Rate Limiter", "FastAPI / Lua scripts\nSmooth Weighted RR", "Port 8000", '#e67e22'),
        (0.70, 0.12, 0.24, 0.26, "Inference Backends\n(vLLM & Cloud)", "vLLM Qwen2.5-32B\nGemini 3.5 / Mistral", "OpenAI API spec", '#2c3e50'),
    ]
    
    for x, y, w, h, name, tech, ports, color in containers:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                     linewidth=2, edgecolor=color, facecolor='white')
        ax.add_patch(rect)
        header = patches.FancyBboxPatch((x, y + h - 0.08), w, 0.08, boxstyle="round,pad=0.015,rounding_size=0.02",
                                       linewidth=0, facecolor=color, alpha=0.15)
        ax.add_patch(header)
        
        ax.text(x + w/2, y + h - 0.04, name, horizontalalignment='center',
                verticalalignment='center', fontsize=9.5, fontweight='bold', color=color)
        ax.text(x + w/2, y + 0.09, tech, horizontalalignment='center',
                verticalalignment='center', fontsize=8, color='#34495e', linespacing=1.2)
        ax.text(x + w/2, y + 0.03, ports, horizontalalignment='center',
                verticalalignment='center', fontsize=7.5, fontweight='bold', color='#7f8c8d')
    
    # GAMA <-> Orchestrator
    ax.annotate('', xy=(0.38, 0.71), xytext=(0.32, 0.71), 
                arrowprops=dict(arrowstyle='<->', color=PALETTE['dark'], lw=2))
    ax.text(0.35, 0.74, "WebSocket", fontsize=7.5, fontweight='bold', ha='center', color=PALETTE['dark'],
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
    
    # Orchestrator <-> Redis
    ax.annotate('', xy=(0.70, 0.71), xytext=(0.64, 0.71), 
                arrowprops=dict(arrowstyle='<->', color=PALETTE['dark'], lw=2))
    ax.text(0.67, 0.74, "Pub/Sub + Cache", fontsize=7.5, fontweight='bold', ha='center', color=PALETTE['dark'],
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
    
    # Orchestrator -> Multi-Modal Routers
    ax.annotate('', xy=(0.20, 0.38), xytext=(0.40, 0.58), 
                arrowprops=dict(arrowstyle='->', color=PALETTE['purple'], lw=1.8))
    ax.text(0.28, 0.50, "HTTP Itinerary Queries", fontsize=7.5, fontweight='bold', color=PALETTE['purple'],
            rotation=38, ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
    
    # Orchestrator -> SWRR Gateway
    ax.annotate('', xy=(0.51, 0.38), xytext=(0.51, 0.58), 
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
    ax.text(0.51, 0.48, "Choice Queries", fontsize=7.5, fontweight='bold', color='#e67e22',
            ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
    
    # SWRR <-> Redis (Rate Limiter)
    ax.annotate('', xy=(0.73, 0.58), xytext=(0.60, 0.38), 
                arrowprops=dict(arrowstyle='<->', color=PALETTE['danger'], lw=1.5, linestyle='--'))
    ax.text(0.69, 0.46, "Lua RPM/TPM", fontsize=7.5, fontweight='bold', color=PALETTE['danger'],
            rotation=45, ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
    
    # SWRR -> Backends
    ax.annotate('', xy=(0.70, 0.25), xytext=(0.64, 0.25), 
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.text(0.67, 0.28, "OpenAI API", fontsize=7.5, fontweight='bold', ha='center', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    ax.set_title('Distributed Microservice Architecture (C4 Container Topology)', fontsize=12.5, weight='bold', pad=15)
    
    out_path = OUTPUT_DIR / "fig_c4_container_topology.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 5. FIG: ASYNCHRONOUS BACKPRESSURE & EDF HOLD (Chapter 15 - Architecture)
# ==============================================================================
def generer_backpressure_curve():
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    
    N = 1000
    n = np.linspace(0, 1000, 500)
    k = 1.5  # settings.world.min_internal_coeff_k
    cap = 30.0
    sans_frein = 20
    
    delta_t = np.zeros_like(n)
    for i, count in enumerate(n):
        if count <= sans_frein:
            delta_t[i] = 0.0
        else:
            delta_t[i] = cap * ((count / N) ** k)
            
    color = PALETTE['primary']
    ax1.set_xlabel('Pending In-Flight Decision Backlog ($n$ queued agents on $N=1{,}000$)')
    ax1.set_ylabel(r'Injected Synchronization Delay $\Delta t$ (seconds)', color=color)
    ax1.plot(n, delta_t, color=color, linewidth=2.5, label=r'Reactive Delay: $\Delta t = 30 \cdot (n/1000)^{1.5}$')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax1.scatter([20], [0], color=PALETTE['success'], s=60, zorder=5)
    ax1.annotate(r'Threshold Floor: $n \leq 20$' + '\n(Zero Throttling)', xy=(20, 0), xytext=(50, 4),
                arrowprops=dict(arrowstyle='->', color=PALETTE['success'], lw=1.2),
                fontweight='bold', color=PALETTE['success'])
    
    ax1.scatter([500], [cap * ((0.5)**k)], color=PALETTE['secondary'], s=60, zorder=5)
    ax1.annotate(r'50% Queue: $\Delta t = 10.6$ s', xy=(500, cap * ((0.5)**k)), xytext=(230, 15),
                arrowprops=dict(arrowstyle='->', color=PALETTE['secondary'], lw=1.2),
                fontweight='bold', color=PALETTE['secondary'])
    
    ax1.scatter([890], [cap * ((0.89)**k)], color=PALETTE['danger'], s=60, zorder=5)
    ax1.annotate(r'89% Queue: $\Delta t = 25.2$ s', xy=(890, cap * ((0.89)**k)), xytext=(560, 28),
                arrowprops=dict(arrowstyle='->', color=PALETTE['danger'], lw=1.2),
                fontweight='bold', color=PALETTE['danger'])
    
    ax1.axvspan(900, 1000, alpha=0.15, color=PALETTE['danger'], label='EDF Hard Freeze Zone (Imminent Departures)')
    
    ax1.set_title('Non-Linear Reactive Backpressure and EDF Simulation Throttling', pad=12)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(-0.5, 32)
    ax1.legend(loc='upper left', frameon=True, facecolor='white')
    
    out_path = OUTPUT_DIR / "fig_backpressure_edf.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 6. FIG: METROPOLITAN COST WATERFALL & RATIO 1:50 (Chapter 9 - Economics)
# ==============================================================================
def generer_metropolitan_waterfall():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})
    
    steps = [
        "Unconstrained\nGenerative LLM",
        "Deterministic\nFeasibility Filter",
        "TypeSafe\nRoutine Routing",
        "Caching &\nExact Replay",
        "Three-Tier\nHybrid Cascade"
    ]
    
    bottoms = [0, 140000 - 91000, 49000 - 45800, 3200 - 2820, 0]
    heights = [140000, 91000, 45800, 2820, 380]
    colors = [PALETTE['danger'], PALETTE['success'], PALETTE['primary'], PALETTE['secondary'], PALETTE['success']]
    
    ax1.bar(range(5), heights, bottom=bottoms, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    
    ax1.text(0, 142000, "$140,000", ha='center', va='bottom', fontweight='bold', color=PALETTE['danger'])
    ax1.text(1, bottoms[1] + heights[1]/2, "-$91,000\n(-65%)", ha='center', va='center', color='white', fontweight='bold')
    ax1.text(2, bottoms[2] + heights[2]/2, "-$45,800\n(-93%)", ha='center', va='center', color='white', fontweight='bold')
    ax1.text(3, bottoms[3] + 1500, "-$2,820", ha='center', va='bottom', color='#2c3e50', fontsize=8)
    ax1.text(4, 1500, "< $400 / day", ha='center', va='bottom', fontweight='bold', color=PALETTE['success'])
    
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(steps, fontsize=8.5)
    ax1.set_ylabel('Simulated Daily Cost (USD per Metropolitan Day)')
    ax1.set_title('Metropolitan Simulation Scaling Law (1.32M Residents)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 160000)
    
    models = ['Generative Agent\n(Gemini 3.5)', 'Typed Classifier\n(Jev-1.13.0)']
    costs = [49.28, 1.06]
    ax2.bar(models, costs, color=[PALETTE['danger'], PALETTE['success']], width=0.45, edgecolor='black', linewidth=0.8)
    
    ax2.text(0, 50.5, "$49.28", ha='center', va='bottom', fontweight='bold', color=PALETTE['danger'])
    ax2.text(1, 2.5, "$1.06\n(1:50 Cost Ratio)", ha='center', va='bottom', fontweight='bold', color=PALETTE['success'])
    
    ax2.set_ylabel('Inference Cost across 23,026 Choices (USD)')
    ax2.set_title('Standard Inference Cost for 23,026 Trips', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 60)
    
    plt.suptitle('Computational Economics and Sovereign Scaling Architecture', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "fig_metropolitan_cost_waterfall.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 7. FIG: FINITE-SAMPLE BIAS & EMD EXPLANATION (Chapter 3 - Metrics)
# ==============================================================================
def generer_metrics_explanation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    x = np.linspace(0, 25, 200)
    cdf_ref = 1 - np.exp(-0.25 * x)
    cdf_model = 1 - np.exp(-0.18 * x)
    
    ax1.plot(x, cdf_ref, color=PALETTE['dark'], linewidth=2.5, label=r'Empirical Target $F_{\mathrm{ref}}(x)$')
    ax1.plot(x, cdf_model, color=PALETTE['danger'], linewidth=2, linestyle='--', label=r'Model Output $F_{\mathrm{model}}(x)$')
    ax1.fill_between(x, cdf_ref, cdf_model, color=PALETTE['primary'], alpha=0.25, 
                     label=r'1D EMD Area $\int |F - G|\,dx$')
    
    ax1.set_xlabel('Trip Travel Time (minutes)')
    ax1.set_ylabel('Cumulative Distribution Function (CDF)')
    ax1.set_title('1D Earth Mover\'s Distance on Continuous Profiles', fontsize=10.5, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True, facecolor='white')
    ax1.set_xlim(0, 25)
    ax1.set_ylim(0, 1.05)
    
    N_samples = np.array([100, 250, 500, 1000, 2000, 3154, 5000, 10000])
    bias = 5.02 * np.sqrt(3154 / N_samples)
    
    ax2.plot(N_samples, bias, marker='o', color=PALETTE['purple'], linewidth=2, markersize=6)
    ax2.axvline(x=3154, color=PALETTE['danger'], linestyle=':', label='Our Sealed Cohort ($N=3{,}154$, Bias = $+5.02$ pt)')
    ax2.axhline(y=5.02, color=PALETTE['danger'], linestyle=':')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Sample Size $N$ (log scale)')
    ax2.set_ylabel(r'Finite-Sample Bias $\mathbb{E}[\mathcal{C}_N] - \mathcal{C}_\infty$ (pt)')
    ax2.set_title('Distributional Metric Bias as Function of Sample Size', fontsize=10.5, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, facecolor='white')
    ax2.set_ylim(0, 30)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig_emd_jsd_explanation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 8. FIG: CONFUSION MATRIX AUDIT (Chapter 14 - Individual Audit)
# ==============================================================================
def generer_confusion_matrices():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    modes = ['Car', 'Walking', 'Transit', 'Bicycle']
    
    cm_gemini = np.array([
        [0.86, 0.05, 0.06, 0.03],
        [0.18, 0.65, 0.12, 0.05],
        [0.22, 0.11, 0.58, 0.09],
        [0.28, 0.08, 0.10, 0.54],
    ])
    
    cm_jev = np.array([
        [0.89, 0.04, 0.05, 0.02],
        [0.15, 0.69, 0.11, 0.05],
        [0.20, 0.10, 0.63, 0.07],
        [0.24, 0.09, 0.11, 0.56],
    ])
    
    im1 = ax1.imshow(cm_gemini, cmap='Blues', vmin=0, vmax=1.0)
    ax1.set_title('Generative LLM (Gemini 3.5 Expert)\nWeighted Accuracy: 67.6%, Cross-Entropy: 0.356', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(modes)
    ax1.set_yticklabels(modes)
    ax1.set_xlabel('Predicted Mode')
    ax1.set_ylabel('Declared Ground Truth Mode')
    
    for i in range(4):
        for j in range(4):
            val = cm_gemini[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax1.text(j, i, f"{val*100:.1f}%", ha='center', va='center', color=color, fontweight='bold')
            
    im2 = ax2.imshow(cm_jev, cmap='Greens', vmin=0, vmax=1.0)
    ax2.set_title('Typed Classifier (Jev-1.13.0 Expert)\nWeighted Accuracy: 70.8%, Cross-Entropy: 0.342', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(modes)
    ax2.set_yticklabels(modes)
    ax2.set_xlabel('Predicted Mode')
    
    for i in range(4):
        for j in range(4):
            val = cm_jev[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax2.text(j, i, f"{val*100:.1f}%", ha='center', va='center', color=color, fontweight='bold')
            
    plt.suptitle('Individual Trip-Level Audit: Confusion Matrices across 9,621 Survey Trips', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "fig_confusion_matrix_audit.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")

# ==============================================================================
# 9. FIG: BILINGUAL TOKEN OVERHEAD (Chapter 10 - Bilingualism)
# ==============================================================================
def generer_bilingual_tokenization():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    blocks = [
        "Person & Household\nDemographics",
        "Trip Geometry &\nSpatial Constraints",
        "Candidate Routes\n& Travel Times",
        "Reasoning Rules &\nTask Instruction",
        "Full Decision\nPrompt Context"
    ]
    
    tokens_en = np.array([125, 88, 164, 185, 562])
    tokens_fr = np.array([162, 114, 204, 229, 709])
    
    x = np.arange(len(blocks))
    width = 0.35
    
    ax.bar(x - width/2, tokens_en, width, label='English Prompt', color=PALETTE['primary'], edgecolor='black', linewidth=0.8)
    ax.bar(x + width/2, tokens_fr, width, label='French Prompt (+26.2% Tokens)', color='#e74c3c', edgecolor='black', linewidth=0.8)
    
    for i in range(len(blocks)):
        diff = ((tokens_fr[i] - tokens_en[i]) / tokens_en[i]) * 100
        ax.text(x[i] + width/2, tokens_fr[i] + 12, f"+{diff:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#c0392b')
    
    ax.set_ylabel('BPE Token Count per Decision Block')
    ax.set_title('Cross-Lingual Subword Tokenization Overhead (BPE Byte Fragmentation)', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, fontsize=8.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    ax.set_ylim(0, 820)
    
    out_path = OUTPUT_DIR / "fig_french_english_tokenization.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path.name}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    generer_ebbinghaus_decay()
    generer_retrieval_components()
    generer_memory_pipeline()
    generer_c4_topology()
    generer_backpressure_curve()
    generer_metropolitan_waterfall()
    generer_metrics_explanation()
    generer_confusion_matrices()
    generer_bilingual_tokenization()
    print(f"\n🎉 All figures were generated successfully in {OUTPUT_DIR}")
