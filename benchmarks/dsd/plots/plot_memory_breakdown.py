import matplotlib.pyplot as plt
import numpy as np

# Data
batch_sizes = [1, 4, 8, 16, 32]
target_model_memory = [14, 14, 14, 14, 14]
target_kv = [0.125, 0.5, 1, 2, 4]
draft_model_memory = [0.3125, 0.3125, 0.3125, 0.3125, 0.3125]
draft_kv = [0.008789, 0.035156, 0.070312, 0.140625, 0.28125]

# Convert to percentages
totals = np.array([sum(x) for x in zip(target_model_memory, target_kv, draft_model_memory, draft_kv)])
target_model_memory_pct = np.array(target_model_memory) / totals * 100
target_kv_pct = np.array(target_kv) / totals * 100
draft_model_memory_pct = np.array(draft_model_memory) / totals * 100
draft_kv_pct = np.array(draft_kv) / totals * 100

# Create figure with larger size
plt.figure(figsize=(7, 5), dpi=300)

# Create custom x positions with equal spacing
x_positions = np.arange(len(batch_sizes))

# Define colors matching the breakdown graph
colors = {
    'target_model': '#2196F3',    # Blue
    'target_kv': '#FF9800',       # Orange
    'draft_model': '#4CAF50',     # Green
    'draft_kv': '#E91E63'         # Pink
}

# Plot stacked bars with consistent width
width = 0.6
plt.bar(x_positions, target_model_memory_pct, width=width, 
        label='Target model', color=colors['target_model'],
        edgecolor='black', linewidth=1)
plt.bar(x_positions, target_kv_pct, width=width, 
        bottom=target_model_memory_pct, label='Target KV', 
        color=colors['target_kv'],
        edgecolor='black', linewidth=1)
plt.bar(x_positions, draft_model_memory_pct, width=width,
        bottom=target_model_memory_pct + target_kv_pct, 
        label='Draft model', color=colors['draft_model'],
        edgecolor='black', linewidth=1)
plt.bar(x_positions, draft_kv_pct, width=width,
        bottom=target_model_memory_pct + target_kv_pct + draft_model_memory_pct, 
        label='Draft KV', color=colors['draft_kv'],
        edgecolor='black', linewidth=1)

# Customize the chart with larger fonts
plt.xlabel('Batch Size', fontsize=22, fontweight='bold', labelpad=0)
plt.ylabel('Memory Usage (%)', fontsize=22, fontweight='bold', labelpad=0)

# Customize grid
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.gca().set_axisbelow(True)

# Set custom x-axis ticks and labels with larger fonts
plt.xticks(x_positions, batch_sizes, fontsize=18, fontweight='bold')
plt.yticks(np.arange(0, 101, 20), 
          [f'{i}%' for i in range(0, 101, 20)], 
          fontsize=18, fontweight='bold')

# Add border to the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

# Customize legend with larger font
plt.legend(bbox_to_anchor=(0.5, 1.15), 
          loc='center', 
          ncol=2, 
          fontsize=16,
          frameon=True,
          edgecolor='grey')

# Set axis limits
plt.xlim(-0.5, len(batch_sizes) - 0.5)
plt.ylim(0, 100)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('benchmarks/dsd/figures/memory_usage.pdf', bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.close()
