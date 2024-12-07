import matplotlib.pyplot as plt
import numpy as np

# Process data
acc_rates = [0.7, 0.8, 0.9]
spec_tokens = [1, 3, 5, 7, 'DSD']  # Added DSD
original_tpt = 1726.7732153825004

# Organize MQA throughput data
mqa_tpt = {
    1: [1731.16, 1762.52, 1821.42],
    3: [1961.26, 2040.47, 2250.25],
    5: [1966.19, 2124.91, 2363.84],
    7: [1863.06, 2089.55, 2396.82],
    'DSD': [1926.53, 2062.29, 2320.81]
}

# Set up the figure
plt.figure(figsize=(6, 3), dpi=300)

# Calculate bar positions
x = np.arange(len(acc_rates))
width = 0.15

# Colors for different methods
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']  # Added color for DSD

# Plot bars for each method
for i, k in enumerate(spec_tokens):
    offset = (i - 2) * width  # Adjusted offset for 5 bars
    if k == 'DSD':
        # Special styling for DSD
        plt.bar(x + offset, mqa_tpt[k], width,
                label='DSD', color=colors[i],
                edgecolor='black', linewidth=1,
                hatch='//')  # Added hatching for DSD
    else:
        plt.bar(x + offset, mqa_tpt[k], width,
                label=f'k={k}', color=colors[i],
                edgecolor='black', linewidth=1)

# Add baseline
plt.axhline(y=original_tpt, color='red', linestyle='--', 
           linewidth=2.5, label='Baseline')

# Customize the plot
plt.xlabel('Acceptance Rate', fontsize=14, fontweight='bold', labelpad=0)
plt.ylabel('Throughput (tokens/s)', fontsize=14, fontweight='bold', labelpad=0)

# Set ticks
plt.xticks(x, [f'{rate:.1f}' for rate in acc_rates], 
           fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Add grid
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.gca().set_axisbelow(True)

# # Add legend
# plt.legend(fontsize=20, frameon=True, 
#           edgecolor='black', ncol=6,
#           bbox_to_anchor=(0.5, 1.15), loc='center')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('throughput_bars.pdf', 
            bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.close()
