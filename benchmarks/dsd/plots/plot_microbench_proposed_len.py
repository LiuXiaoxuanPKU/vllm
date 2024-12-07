import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

# Set style configurations
# plt.style.use('seaborn')
matplotlib.rcParams['font.size'] = 18

# Modern color palette
COLORS = {
    0.5: '#2196F3',  # Blue
    0.7: '#FF9800',  # Orange
    0.9: '#4CAF50'   # Green
}

MARKERS = {
    0.5: 'o',  # Circle
    0.7: 's',  # Square
    0.9: '^'   # Triangle
}

def load(filename):
    """Load trace data from JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return data['traces']

def load_all(acc, input_len, max_k, all_batch_sizes):
    """Load all trace data for different batch sizes."""
    data = {}
    for batch_size in all_batch_sizes:
        data[batch_size] = load(
            f"{tracedir}input={input_len}_{batch_size}_{acc}_True_k={max_k}.json")
    return data

def get_avg_proposed_len(data):
    """Calculate average proposed length from trace data."""
    proposed_lens = [trace['proposed_len'] for trace in data 
                    if trace['type'] != 'Request' and 'proposed_len' in trace]
    return sum(proposed_lens) / (len(proposed_lens) + 1e-5)

def create_legend_figure():
    """Create a separate figure for just the legend."""
    # Create figure for legend
    figlegend = plt.figure(figsize=(10, 0.5))
    ax = figlegend.add_subplot(111)
    
    # Create dummy lines for legend
    lines = []
    labels = []
    
    # Add entries for each acceptance rate
    for acc in [0.5, 0.7, 0.9]:
        # Create a line with marker for the legend
        line = plt.Line2D([0], [0],
                         color=COLORS[acc],
                         marker=MARKERS[acc],
                         markersize=12,
                         linewidth=3,
                         markeredgecolor='black',
                         markeredgewidth=2,
                         label=f'acc={acc}')
        lines.append(line)
        labels.append(f'acc={acc}')
    
    # Create the legend
    ax.legend(lines, labels,
             loc='center',
             ncol=3,
             fontsize=20,
             frameon=True,
             borderaxespad=0)
    
    # Turn off axis
    ax.set_axis_off()
    
    # Save legend figure
    plt.savefig('benchmarks/dsd/figures/proposed_len_legend.pdf',
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.1)
    plt.close()
    
def create_proposed_length_plot():
    """Create enhanced plot with internal legend."""
    all_batch_sizes = [1, 4, 8, 16, 32, 64]
    x_positions = np.arange(len(all_batch_sizes))
    acc_rates = [0.5, 0.7, 0.9]
    input_len = 256
    max_k = 8

    # Create figure
    plt.figure(figsize=(5, 3), dpi=300)
    ax = plt.gca()

    # Plot data
    for acc in acc_rates:
        data = load_all(acc, input_len, max_k, all_batch_sizes)
        proposed_lens = [get_avg_proposed_len(data[batch_size]) 
                        for batch_size in all_batch_sizes]
        
        plt.plot(x_positions, proposed_lens, 
                marker=MARKERS[acc], 
                color=COLORS[acc],
                label=f"acc={acc}",
                linewidth=3,
                markersize=14,
                markeredgecolor='black',
                markeredgewidth=2)

    # X-axis styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_batch_sizes, fontsize=14, fontweight='bold')
    
    # Labels
    plt.xlabel("Batch Size", fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel("Average Proposed Len", fontsize=14, fontweight='bold', labelpad=10)
    
    # Y-axis ticks
    plt.yticks(fontsize=14, fontweight='bold')

    # Grid styling
    plt.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # # Legend inside the figure
    # plt.legend(
    #           ncol=3,             # 3 columns
    #           bbox_to_anchor=(0.5, 1.05), loc='center',  # Anchor to upper right corner
    #           fontsize=20,
    #           framealpha=0.9,      # Slight transparency
    #           edgecolor='black',   # Black edge
    #           frameon=True,
    #           borderpad=0.8,       # Padding inside legend border
    #           handletextpad=0.5)   # Space between handle and text

    # Tighter y-axis limits
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.05)

    # Adjust layout
    plt.tight_layout(pad=0.2)

    # Save figure
    plt.savefig("benchmarks/dsd/figures/h100_proposed_len.pdf",
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    tracedir = "benchmarks/dsd/trace/"
    create_proposed_length_plot()
    create_legend_figure()