
import lovely_tensors as lt
import matplotlib.pyplot as plt
import torch
from as501_modules import load_hex_file_to_tensor
import numpy as np

def summarize_tensor(tensor, filename):
    """Summarize the distribution of a PyTorch tensor."""
    # Convert tensor to numpy array for easier manipulation
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
    plt.hist(flat_tensor, bins=bins, edgecolor='black')
    plt.title("Tensor Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f'output/{filename}_summarize.png')



def plot_heatmap(tensor, filename):
    plt.imshow(tensor, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 색상 막대 추가
    plt.title("Heatmap of Tensor")
    plt.savefig(f'output/{filename}_heatmap.png')


def main():
    filename = 'weight1'  # replace with your file path
    tensor = load_hex_file_to_tensor(f'parameter/{filename}.txt')
    summarize_tensor(tensor, filename)
    #plot_heatmap(tensor, filename)

if __name__ == '__main__':
    main()