
import lovely_tensors as lt
import matplotlib.pyplot as plt
import torch
from as501_modules import load_hex_file_to_tensor
import numpy as np

def summarize_tensor(tensor, filename, quant):
    """Summarize the distribution of a PyTorch tensor."""
    # Convert tensor to numpy array for easier manipulation
    if quant==True:
        tensor = uniform_quantization(tensor, 5)
    tensor_np = tensor.numpy()
    
    # Compute basic statistics
    min_val = torch.min(tensor).item()
    max_val = torch.max(tensor).item()
    mean_val = torch.mean(tensor.float()).item()
    median_val = torch.median(tensor.float()).item()
    std_val = torch.std(tensor.float()).item()
    
    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")
    print(f"Mean value: {mean_val}")
    print(f"Median value: {median_val}")
    print(f"Standard deviation: {std_val}")
    flat_tensor = tensor_np.flatten()
    bins = np.arange(flat_tensor.min(), flat_tensor.max() + 2) - 0.5

    # Plot histogram
    plt.clf()
    plt.hist(flat_tensor, bins=bins, edgecolor='black')
    plt.title("Tensor Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if quant==True:
        plt.savefig(f'output/{filename}_summarize_quant.png')
    else:
        plt.savefig(f'output/{filename}_summarize.png')



def plot_heatmap(tensor, filename):
    plt.clf()
    plt.imshow(tensor, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 색상 막대 추가
    plt.title("Heatmap of Tensor")
    plt.savefig(f'output/{filename}_heatmap.png')


def calculate_sparsity(tensor):
    """Calculate the sparsity of a PyTorch tensor."""
    num_zeros = torch.numel(tensor) - torch.count_nonzero(tensor)
    total_elements = torch.numel(tensor)
    sparsity = num_zeros / total_elements
    return sparsity

def uniform_quantization(tensor, num_bits=4):
    """Perform uniform quantization on a PyTorch tensor."""
    # Convert tensor to float32
    tensor = tensor.float()
    # Calculate the range of the tensor
    max_val = torch.max(tensor).item()
    min_val = torch.min(tensor).item()
    # Calculate the quantization step
    # quant_step is reciprocal of scaling factor
    quant_step = (max_val - min_val) / (2**num_bits - 1)
    # Quantize the tensor
    tensor = torch.round(tensor / quant_step) * quant_step
    return tensor

def main():
    filename = 'weight1'  # replace with your file path
    tensor = load_hex_file_to_tensor(f'parameter/{filename}.txt')
    summarize_tensor(tensor, filename, quant=True)
    #plot_heatmap(tensor, filename)

if __name__ == '__main__':
    main()