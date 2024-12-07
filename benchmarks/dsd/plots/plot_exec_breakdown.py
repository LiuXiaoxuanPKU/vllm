import json
import matplotlib.pyplot as plt
import numpy as np

# Constants and configuration
TRACEDIR = "benchmarks/dsd/trace/"

# Colors for different time components
COLORS = {
    'draft': '#2196F3',    # Blue
    'target': '#FF9800',   # Orange
    'overhead': '#4CAF50'  # Green
}

def get_filename(input_len, batch_size, acc, max_k):
    return f"{TRACEDIR}input={input_len}_{batch_size}_{acc}_False_k={max_k}.json"

def load(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data['traces']

def get_avg_times(times):
    return sum(times) / len(times)

def get_exec_time(data):
    step_traces = [trace for trace in data if trace['type'] == 'Step']
    step_traces = [trace for trace in step_traces if 'measured_draft_time' in trace]
    draft_times = [trace['measured_draft_time'] for trace in step_traces]
    target_times = [trace['measured_target_time'] for trace in step_traces]
    overhead_times = [trace['measured_overhead_time'] for trace in step_traces]
    return get_avg_times(draft_times), get_avg_times(target_times), get_avg_times(overhead_times)

def create_percentage_breakdown_plot(all_times, batch_sizes, k_values):
    """Create stacked bar plot showing percentage breakdown of time components."""
    plt.figure(figsize=(8, 4), dpi=300)
    ax = plt.gca()
    
    # Calculate bar positions
    num_groups = len(batch_sizes)
    group_width = 0.8
    bar_width = group_width / len(k_values)
    
    # Create positions for each group of bars
    group_positions = np.arange(num_groups)
    
    # Create legend handles
    legend_handles = []
    legend_labels = []
    
    # Plot bars for each k value
    for i, k in enumerate(k_values):
        # Calculate x positions for this k value
        x_positions = group_positions + (i - len(k_values)/2 + 0.5) * bar_width
        
        # Calculate percentages for each component
        percentages = []
        for bs in batch_sizes:
            total_time = (all_times[k][bs]['draft'] + 
                         all_times[k][bs]['target'] + 
                         all_times[k][bs]['overhead'])
            percentages.append({
                'draft': (all_times[k][bs]['draft'] / total_time) * 100,
                'target': (all_times[k][bs]['target'] / total_time) * 100,
                'overhead': (all_times[k][bs]['overhead'] / total_time) * 100
            })
        
        # Extract percentage arrays
        draft_pcts = [p['draft'] for p in percentages]
        target_pcts = [p['target'] for p in percentages]
        overhead_pcts = [p['overhead'] for p in percentages]
        
        # Create stacked bars
        draft_bars = plt.bar(x_positions, draft_pcts, bar_width, 
                           color=COLORS['draft'], edgecolor='black', linewidth=1)
        target_bars = plt.bar(x_positions, target_pcts, bar_width, 
                            bottom=draft_pcts,
                            color=COLORS['target'], edgecolor='black', linewidth=1)
        overhead_bars = plt.bar(x_positions, overhead_pcts, bar_width, 
                              bottom=[sum(x) for x in zip(draft_pcts, target_pcts)],
                              color=COLORS['overhead'], edgecolor='black', linewidth=1)
        
        # Add to legend for first k value only
        if i == 0:
            legend_handles.extend([draft_bars, target_bars, overhead_bars])
            legend_labels.extend(['Draft', 'Target', 'Overhead'])
        
        # Add k-value labels
        for j, x_pos in enumerate(x_positions):
            plt.text(x_pos, -4, f'{k}', ha='center', va='top',
                    fontsize=17)

    # Customize the plot
    plt.xlabel('Batch Size', fontsize=19, fontweight='bold', labelpad=5)
    plt.ylabel('Ratio of Total Time (%)', fontsize=19, fontweight='bold', labelpad=0)
    
    # Set x-axis ticks and labels
    plt.xticks(group_positions, batch_sizes, fontsize=17, fontweight='bold')
    ax.tick_params(axis='x', pad=15) 
    plt.yticks(np.arange(0, 101, 20), fontsize=15, fontweight='bold')
    
    # Add grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.gca().set_axisbelow(True)
    
    # Add legend
    plt.legend(legend_handles, legend_labels, 
              fontsize=15, ncol=3, bbox_to_anchor=(0.5, 1.15),
              loc='center', frameon=True, edgecolor='black')
    
    # Set y-axis limits to accommodate labels and show full percentage range
    plt.ylim(-3, 100)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('benchmarks/dsd/figures/time_percentage_breakdown.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Parameters
    acc_rates = [0.7]
    input_len = 256
    k_values = [1, 3, 5, 7]
    batch_sizes = [1, 4, 8, 16, 32]
    
    # Initialize dictionary to store all times
    all_times = {k: {bs: {'draft': 0, 'target': 0, 'overhead': 0} 
                     for bs in batch_sizes} for k in k_values}
    
    # Collect data
    for k in k_values:
        for bs in batch_sizes:
            data = load(get_filename(input_len, bs, acc_rates[0], k))
            draft_time, target_time, overhead_time = get_exec_time(data)
            
            # Store times
            all_times[k][bs]['draft'] = draft_time
            all_times[k][bs]['target'] = target_time
            all_times[k][bs]['overhead'] = overhead_time
    
    # Create visualization
    create_percentage_breakdown_plot(all_times, batch_sizes, k_values)

if __name__ == "__main__":
    main()