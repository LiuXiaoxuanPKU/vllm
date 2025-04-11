import os
import torch
import itertools
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import json
mpl.rcParams['agg.path.chunksize'] = 10000


def print_avg_min_max_std(tensor: torch.Tensor, name: str):
    """Print the average, minimum, maximum, and standard deviation of the tensor"""
    tensor_flat = tensor.flatten().to(torch.float64)
    num = tensor_flat.size(0)
    avg = torch.mean(tensor_flat)
    min_val = torch.min(tensor_flat)
    max_val = torch.max(tensor_flat)
    std = torch.std(tensor_flat)
    
    # Plt text avg, min, max, std on the graph up to 6 decimal places
    if tensor_flat.shape[0] > 2**24:
        sample_idx = torch.randint(0, tensor_flat.shape[0], (2**24,))
        tensor_flat = tensor_flat[sample_idx]
    per_25 = torch.quantile(tensor_flat, 0.25)
    per_50 = torch.quantile(tensor_flat, 0.50)
    per_75 = torch.quantile(tensor_flat, 0.75)
    per_99 = torch.quantile(tensor_flat, 0.99)
    per_01 = torch.quantile(tensor_flat, 0.01)
    per_995 = torch.quantile(tensor_flat, 0.995)
    per_005 = torch.quantile(tensor_flat, 0.005)
    per_999 = torch.quantile(tensor_flat, 0.999)
    per_001 = torch.quantile(tensor_flat, 0.001)
    
    print(f"{name}\n\t Num: {num:.0f}, Avg: {avg:.7f}, Min: {min_val:.7f}, "
          f"Max: {max_val:.7f}, Std: {std:.7f}")
    print(f"\t0.1%: {per_001:.7f}, 0.5%: {per_005:.7f}, 1%: {per_01:.7f}, "
          f"25%: {per_25:.7f}, 50%: {per_50:.7f}, 75%: {per_75:.7f}, "
          f"99%: {per_99:.7f}, 99.5%: {per_995:.7f}, 99.9%: {per_999:.7f}")

def print_avg_min_max_std_histogram(tensor: torch.Tensor, 
                                    name: str, output_dir: str):
    """Print the average, minimum, maximum, and standard deviation of the tensor
    and save the histogram"""
    tensor_flat = tensor.flatten().to(torch.float64)
    num = tensor_flat.size(0)
    avg = torch.mean(tensor_flat)
    min_val = torch.min(tensor_flat)
    max_val = torch.max(tensor_flat)
    std = torch.std(tensor_flat)
    
    # Plot the histogram and save it
    plt.hist(tensor_flat.cpu().numpy(), bins=500)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Plt text avg, min, max, std on the graph up to 6 decimal places
    if tensor_flat.shape[0] > 2**24:
        sample_idx = torch.randint(0, tensor_flat.shape[0], (2**24,))
        tensor_flat = tensor_flat[sample_idx]
    per_25 = torch.quantile(tensor_flat, 0.25)
    per_50 = torch.quantile(tensor_flat, 0.50)
    per_75 = torch.quantile(tensor_flat, 0.75)
    per_99 = torch.quantile(tensor_flat, 0.99)
    per_01 = torch.quantile(tensor_flat, 0.01)
    per_995 = torch.quantile(tensor_flat, 0.995)
    per_005 = torch.quantile(tensor_flat, 0.005)
    per_999 = torch.quantile(tensor_flat, 0.999)
    per_001 = torch.quantile(tensor_flat, 0.001)
    plt.text(0.6, 0.95, f"Num: {num:.0f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.9, f"Avg: {avg:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.85, f"Min: {min_val:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.8, f"Max: {max_val:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.75, f"Std: {std:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    # plt.text(0.85, 0.95, f"0.1%: {per_001:.7f}", ha='center', va='center', 
    #          transform=plt.gca().transAxes)
    # plt.text(0.85, 0.9, f"25%: {per_25:.7f}", ha='center', va='center', 
    #          transform=plt.gca().transAxes)
    # plt.text(0.85, 0.85, f"50%: {per_50:.7f}", ha='center', va='center', 
    #          transform=plt.gca().transAxes)
    # plt.text(0.85, 0.8, f"75%: {per_75:.7f}", ha='center', va='center', 
    #          transform=plt.gca().transAxes)
    # plt.text(0.85, 0.75, f"99%: {per_99:.7f}", ha='center', va='center', 
    #          transform=plt.gca().transAxes)
    # plt.text(0.85, 0.7, f"99.5%: {per_995:.7f}", ha='center', va='center', 
    #          transform=plt.gca().transAxes)
    # plt.text(0.85, 0.65, f"99.9%: {per_999:.7f}", ha='center', va='center', 
    #          transform=plt.gca().transAxes)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{name}.png", dpi=1600)
    plt.show()
    plt.close()

def print_avg_min_max_std_heatmap(tensor: torch.Tensor, 
                                  name: str, output_dir: str):
    """Print the average, minimum, maximum, and standard deviation of the tensor
    and save the heatmap"""
    assert tensor.dim() == 2
    tensor_flat = tensor.flatten().to(torch.float64)
    num = tensor_flat.size(0)
    avg = torch.mean(tensor_flat)
    min_val = torch.min(tensor_flat)
    max_val = torch.max(tensor_flat)
    std = torch.std(tensor_flat)

    # Plot the heatmap and save it
    plt.pcolormesh(tensor.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    if tensor_flat.shape[0] > 2**24:
        sample_idx = torch.randint(0, tensor_flat.shape[0], (2**24,))
        tensor_flat = tensor_flat[sample_idx]
    per_001 = torch.quantile(tensor_flat, 0.001)
    per_25 = torch.quantile(tensor_flat, 0.25)  
    per_50 = torch.quantile(tensor_flat, 0.50)
    per_75 = torch.quantile(tensor_flat, 0.75)
    per_999 = torch.quantile(tensor_flat, 0.999)
    plt.title(f"{name}\nAvg: {avg:.4f}, Min: {min_val:.4f}, Max: {max_val:.4f}, "
              f"Std: {std:.4f}\n0.1%: {per_001:.4f}, 25%: {per_25:.4f}, "
              f"50%: {per_50:.4f}, 75%: {per_75:.4f}, 99.9%: {per_999:.4f}")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{name}.png", dpi=1600)
    plt.show()
    plt.close()

def plot_3d(tensor: torch.Tensor, 
            name: str, output_dir: str, 
            scale: float = None, 
            color_id: torch.tensor = None, 
            centroids: torch.tensor = None):
    """Plot 3-D tensor data as a graph, with X axis as tensor[:,0], Y axis as
    tensor[:,1] and Z axis as tensor[:,2]"""
    assert tensor.dim() == 2
    if not tensor.size(1) == 3:
        return
    tensor_cpu = tensor.to(torch.float64).cpu()
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if scale is not None:
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)
    if color_id is not None:
        ax.scatter(tensor_cpu[:,0].cpu().numpy(), 
                   tensor_cpu[:,1].cpu().numpy(), 
                   tensor_cpu[:,2].cpu().numpy(),
                   c=color_id.cpu().numpy(), cmap='gist_rainbow', s=0.3)
    else:
        ax.scatter(tensor_cpu[:,0].cpu().numpy(), 
                   tensor_cpu[:,1].cpu().numpy(), 
                   tensor_cpu[:,2].cpu().numpy(), s=0.1)
    if centroids is not None:
        ax.scatter(centroids[:,0].cpu().numpy(), 
                   centroids[:,1].cpu().numpy(), 
                   centroids[:,2].cpu().numpy(), c='black', s=0.4)
    plt.title(name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{name}.png", dpi=1600)
    plt.show()
    plt.close()
    
def plot_2d(tensor: torch.Tensor, 
            name: str, output_dir: str, 
            scale: float = None, 
            color_id: torch.tensor = None, 
            centroids: torch.tensor = None):
    """Plot 2-D tensor data as a graph, with X axis as tensor[:,0] and Y axis
    as tensor[:,1]"""
    assert tensor.dim() == 2
    if not tensor.size(1) == 2:
        return
    tensor_cpu = tensor.to(torch.float64).cpu()
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if scale is not None:
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
    if color_id is not None:
        ax.scatter(tensor_cpu[:,0].cpu().numpy(), 
                   tensor_cpu[:,1].cpu().numpy(),
                    c=color_id.cpu().numpy(), cmap='gist_rainbow', s=0.1)
    else:
        ax.scatter(tensor_cpu[:,0].cpu().numpy(), 
                   tensor_cpu[:,1].cpu().numpy(), s=0.1)
    if centroids is not None:
        ax.scatter(centroids[:,0].cpu().numpy(), 
                   centroids[:,1].cpu().numpy(), c='black', s=1)
        # for i in range(centroids.size(0)):
        #     ax.text(centroids[i,0].cpu().numpy() - 0.025, 
        #             centroids[i,1].cpu().numpy() + 0.075, f"{i}", fontsize=4)
    plt.title(name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{name}.png", dpi=1600)
    plt.show()
    plt.close()


def plot_1d(tensor: torch.Tensor, name: str, output_dir: str):
    """Plot 1-D tensor data as a graph, with X axis as tensor index and Y axis 
    as tensor value"""
    assert tensor.dim() == 1
    tensor_cpu = tensor.to(torch.float64).cpu()
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.plot(tensor_cpu.cpu().numpy(), lw=0.5)
    plt.title(name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{name}.png", dpi=1600)
    plt.show()
    plt.close()
    
def plot_bar(tensor: torch.Tensor, name: str, output_dir: str, caption=None):
    """Plot 1-D tensor data as a bar graph, with X axis as tensor index and Y 
    axis as tensor value"""
    assert tensor.dim() == 1
    tensor_cpu = tensor.to(torch.float64).cpu()
    fig = plt.figure(dpi=800, figsize=(16, 6))  # Increase the figure size for wider output
    ax = fig.add_subplot(111)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.bar(range(tensor_cpu.size(0)), tensor_cpu.cpu().numpy())
    # Add caption per bar
    if caption is not None:
        for i in range(tensor_cpu.size(0)):
            ax.text(i, tensor_cpu[i].item(), f"{caption[i].item()}",
                    ha='center', va='bottom', fontsize=8)
    plt.title(name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{name}.png", dpi=1200)
    plt.show()
    plt.close()
    
def export_topk_per_layer_as_csv(topk_per_layer, path: str):
    num_layer = len(topk_per_layer)
    num_tokens = len(topk_per_layer[0])
    num_topk = len(topk_per_layer[0][0])
    with open(path, 'w') as f:
        f.write(
            f'Num layers: {num_layer}, Num tokens: {num_tokens}, '
            f'Num top k: {num_topk}\n'
        )
        f.write('layer, token,')
        for j in range(num_topk):
            f.write(f'top_{j},')
        f.write('\n')
        for i in range(num_layer):
            for j in range(num_tokens):
                f.write(f'{i},{j},')
                for k in range(num_topk):
                    f.write(f'{topk_per_layer[i][j][k]},')
                f.write('\n')
                
def export_model_dataset_histogram(data_list,
                                   output_dir: str,
                                   filename: str,
                                   xlabel: str,
                                   ylabel: str,
                                   title: str):
    plt.figure(figsize=(10, 6))
    for entry in data_list:
        label = f"{entry['model_name']}/{entry['dataset_name']}"
        values = entry['data']
        plt.hist(values, bins=30, alpha=0.5, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{filename}.png", dpi=1200)
    plt.show()
    plt.close()

output_path = "acceptance_rate_parse_output/"
path = sys.argv[1]
num_acceptance_data_list = []
avg_acceptance_data_list = []
for filename in os.listdir(path):
    if filename.endswith(".json"):
        with open(os.path.join(path, filename), 'r') as f:
            data = json.load(f)
            model_name = data['model'].split('/')[-1]
            dataset_name = data['dataset']
            if dataset_name == "hf":
                dataset_name = data['dataset_path'].split('/')[-1]
            acceptance_rates_per_req = data['acceptance_rates_per_req']
            num_acceptance_per_req = []
            avg_acceprance_per_req = []
            for req_idx, acceptance_rate in acceptance_rates_per_req.items():
                num_acceptance_vals = len(acceptance_rate)
                avg_acceptance_rate = sum(acceptance_rate)/num_acceptance_vals
                num_acceptance_per_req.append(num_acceptance_vals)
                avg_acceprance_per_req.append(avg_acceptance_rate)
            num_acceptance_data_list.append({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'data': num_acceptance_per_req,
            })
            avg_acceptance_data_list.append({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'data': avg_acceprance_per_req,
            })

# Create histogram for num_acceptance_per_req for each model/dataset pair.
export_model_dataset_histogram(
    num_acceptance_data_list,
    output_dir=output_path,
    filename="num_acceptance_per_req_histogram",
    xlabel="Number of Acceptance Rates per Request",
    ylabel="Frequency",
    title="Histogram of Number of Acceptance Rates per Request by Model/Dataset Pair"
)

# Create histogram for avg_acceptance_per_req for each model/dataset pair.
export_model_dataset_histogram(
    avg_acceptance_data_list,
    output_dir=output_path,
    filename="avg_acceptance_per_req_histogram",
    xlabel="Average Acceptance Rate",
    ylabel="Frequency",
    title="Histogram of Average Acceptance Rate by Model/Dataset Pair"
)
unique_models = set(item['model_name'] for item in num_acceptance_data_list)
for model in unique_models:
    model_data = [item for item in num_acceptance_data_list if item['model_name'] == model]
    model_avg_data = [item for item in avg_acceptance_data_list if item['model_name'] == model]
    if len(model_data) > 0:
        export_model_dataset_histogram(
            model_data,
            output_dir=output_path,
            filename=f"num_acceptance_per_req_histogram_{model}",
            xlabel="Number of Acceptance Rates per Request",
            ylabel="Frequency",
            title=f"Histogram of Number of Acceptance Rates per Request for {model}"
        )
    if len(model_avg_data) > 0:
        export_model_dataset_histogram(
            model_avg_data,
            output_dir=output_path,
            filename=f"avg_acceptance_per_req_histogram_{model}",
            xlabel="Average Acceptance Rate",
            ylabel="Frequency",
            title=f"Histogram of Average Acceptance Rate for {model}"
        )
