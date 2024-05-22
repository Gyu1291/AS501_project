import torch

def hex_to_int(hex_str):
    """Convert a hex string to an integer."""
    #print(hex_str)
    last_two_hex = hex_str[-2:]
    num = int(last_two_hex, 16)
    #check sign bit
    if num & 0x80:
        num -= 256
        
    return num

def load_hex_file_to_tensor(file_path):
    """Load a text file of 8-bit hex numbers and convert it to a PyTorch tensor."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Remove any whitespace characters like \n at the end of each line
    lines = [line.strip() for line in lines]
    
    # Convert hex strings to integers
    int_values = [hex_to_int(line) for line in lines]
    
    # Convert list of integers to PyTorch tensor
    tensor = torch.tensor(int_values, dtype=torch.float32)
    if file_path == 'parameter/weight1.txt':
        tensor = tensor.reshape(128, 784)
    elif file_path == 'parameter/weight2.txt':
        tensor = tensor.reshape(64, 128)
    elif file_path == 'parameter/weight3.txt':
        tensor = tensor.reshape(10, 64)
    return tensor






# # Example usage
# filename = 'weight3'  # replace with your file path
# file_path = filename + '.txt'
# lt.monkey_patch()
# tensor = load_hex_file_to_tensor(file_path)

# #tensor.plt.fig.savefig("weight3.png")
# summarize_tensor(tensor, filename)
# #plot_heatmap(tensor, filename)
