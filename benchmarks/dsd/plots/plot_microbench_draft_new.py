import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style configurations
# plt.style.use('seaborn')
matplotlib.rcParams['font.size'] = 18  # Increased base font size

# Constants
MODEL = "llama2-7b"
MODE = "draft"
ROOT = "benchmarks/dsd/results/"
INPUT_LEN = 256
ALL_BATCH_SIZES = [1, 4, 8, 16, 32, 64]
KS = [1, 3, 5, 7]

# Distinct color palette
COLORS = {
    1: '#2196F3',    # Bright Blue
    3: '#FF9800',    # Bright Orange
    5: '#4CAF50',    # Bright Green
    7: '#E91E63',    # Pink
    'dsd': '#9C27B0'  # Purple
}

def load_data(model, mode, bz, input_len, method, acc=None, k=None):
    """Load benchmark data from JSON files."""
    data_dir = f"{ROOT}/{model}-{mode}"
    if method == 'org':
        filename = f"{data_dir}/{method}_bz={bz}_input-len={input_len}.json"
    elif method == 'dsd':
        filename = f"{data_dir}/{method}_bz={bz}_input-len={input_len}_acc={acc}.json"
    else:
        filename = f"{data_dir}/{method}={k}_bz={bz}_input-len={input_len}_acc={acc}.json"
    
    if not os.path.exists(filename):
        return {"avg_latency": 0}
    
    with open(filename, 'r') as f:
        return json.load(f)

def load_all(acc_rates):
    """Load all benchmark data for different methods and configurations."""
    data = {}
    for method in ['dsd', 'org', 'vsd']:
        data[method] = {}
        for bz in ALL_BATCH_SIZES:
            if method == 'org':
                data[method][bz] = load_data(MODEL, MODE, bz, INPUT_LEN, method)
                continue
            data[method][bz] = {}
            for acc in acc_rates:
                if method == "dsd":
                    data[method][bz][acc] = load_data(MODEL, MODE, bz, INPUT_LEN, method, acc)
                    continue
                data[method][bz][acc] = {}
                for k in KS:
                    data[method][bz][acc][k] = load_data(MODEL, MODE, bz, INPUT_LEN, method, acc, k)
    return data

def create_legend_figure():
    """Create a separate figure for the legend."""
    figlegend = plt.figure(figsize=(10, 0.5))
    ax = figlegend.add_subplot(111)
    
    # Create dummy plots to generate legend handles
    handles = []
    labels = []
    
    # Add VSD bars
    for k in KS:
        handle = plt.Rectangle((0,0), 1, 1, fc=COLORS[k], ec='black', linewidth=1.5)
        handles.append(handle)
        labels.append(f'k={k}')
    
    # Add DSD bar
    handle = plt.Rectangle((0,0), 1, 1, fc=COLORS['dsd'], ec='black', linewidth=1.5, hatch='//')
    handles.append(handle)
    labels.append('DSD')
    
    # Add baseline
    handle = plt.Line2D([0], [0], color='#DC3545', linestyle='--', linewidth=2.5)
    handles.append(handle)
    labels.append('Baseline')
    
    # Create legend
    ax.legend(handles, labels, loc='center', ncol=6, fontsize=20, frameon=True)
    ax.axis('off')
    
    # Save legend figure
    plt.savefig(f"benchmarks/dsd/figures/micro_legend.pdf", 
                bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_speedup_bar(data, acc):
    """Create an enhanced bar plot with much larger text."""
    # Figure setup with higher resolution and larger size
    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.gca()
    
    x = np.arange(len(ALL_BATCH_SIZES))
    width = 0.15
    
    # Plot VSDs with enhanced styling
    for k in KS:
        vsd_speedups = []
        for bz in ALL_BATCH_SIZES:
            org = data['org'][bz]
            vsd = data['vsd'][bz][acc][k]
            if vsd['avg_latency'] == 0:
                vsd_speedups.append(0)
                continue
            vsd_speedups.append(org['avg_latency'] / vsd['avg_latency'])
        
        ax.bar(x + (k - 3) * width / 2, vsd_speedups, width, 
               label=f'k={k}', color=COLORS[k],
               edgecolor='black', linewidth=1.5)

    # Plot DSD with enhanced styling
    dsd_speedups = []
    for bz in ALL_BATCH_SIZES:
        org = data['org'][bz]
        dsd = data['dsd'][bz][acc]
        dsd_speedups.append(org['avg_req_latency'] / dsd['avg_req_latency'])

    ax.bar(x + width * 3, dsd_speedups, width, label='DSD', 
           color=COLORS['dsd'], hatch='//',
           edgecolor='black', linewidth=1.5)

    # Enhanced styling with much larger text
    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels(ALL_BATCH_SIZES, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Reference line with enhanced styling
    plt.axhline(y=1.0, color='#DC3545', linestyle='--', 
                label='Baseline', linewidth=2.5)

    # Labels and title with much larger text
    plt.xlabel('Batch Size', fontsize=24, fontweight='bold', labelpad=15)
    if acc == 0.5:
        plt.ylabel('Speedup', fontsize=24, fontweight='bold', labelpad=15)
    # plt.title(f'Performance Speedup (acc={acc})', 
    #           fontsize=26, fontweight='bold', pad=20)

    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # # Legend styling with larger text
    # plt.legend(bbox_to_anchor=(0.5, 1.05), loc='center', 
    #           ncol=6, fontsize=16, frameon=True,
    #           borderaxespad=0)

    # Y-axis formatting with larger numbers
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    
    # Adjust layout
    plt.tight_layout()

    # Save with high quality
    plt.savefig(
        f"benchmarks/dsd/figures/h100_bar_7b_draft_speedup_acc={acc}.pdf",
        bbox_inches='tight', dpi=300
    )
    plt.close()

if __name__ == "__main__":
    acc_rates = [0.5, 0.7, 0.9]
    data = load_all(acc_rates)
    for acc in acc_rates:
        plot_speedup_bar(data, acc)
    create_legend_figure()