from torch import nn
import torch
from as501_modules import load_hex_file_to_tensor, hex_to_int
from analyze_tensor import plot_heatmap, summarize_tensor, calculate_sparsity


def load_vector(file_path):
    """Load a text file of 8-bit hex numbers and convert it to a PyTorch tensor."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Remove any whitespace characters like \n at the end of each line
    lines = [line.strip() for line in lines]
    
    # Convert hex strings to integers
    int_values = [hex_to_int(line) for line in lines]
    int_values = torch.tensor(int_values, dtype=torch.float32)
    
    mean = int_values.mean()
    std = int_values.std()
    normalized_tensor = (int_values - mean) / std
    
    return normalized_tensor


def read_parameter():
    weight_dict = {}
    bias_dict = {}
    weight_dict['fc1'] = load_vector('parameter/weight1.txt')
    weight_dict['fc2'] = load_vector('parameter/weight2.txt')
    weight_dict['fc3'] = load_vector('parameter/weight3.txt')

    bias_dict['fc1'] = load_vector('parameter/bias1.txt')
    bias_dict['fc2'] = load_vector('parameter/bias2.txt')
    bias_dict['fc3'] = load_vector('parameter/bias3.txt')
    return weight_dict, bias_dict


def read_input(file_name):
    tensor = load_hex_file_to_tensor(file_name)
    # Reshape the tensor to (1, 784)
    tensor = tensor.reshape(-1, 784)
    return tensor

def fully_connected(input, weight, bias, output_size, input_size):
    output = torch.zeros(output_size)
    for i in range(output_size):
        for j in range(input_size):
            output[i] += input[j] * weight[i*input_size+j]
        output[i] += bias[i]
        if output[i] < 0:
            output[i] = 0
    return output


def main():
    weight_dict, bias_dict = read_parameter()
    batch_input = read_input('image/image_10.txt')
    output_list = []

    #check image
    #plot_heatmap(batch_input[0].to('cpu').reshape(28, 28), 'sample_image_heatmap')

    for single_input in batch_input:
        x = fully_connected(single_input, weight_dict['fc1'], bias_dict['fc1'], 128, 784)
        x = fully_connected(x, weight_dict['fc2'], bias_dict['fc2'], 64, 128)
        output = fully_connected(x, weight_dict['fc3'], bias_dict['fc3'], 10, 64)
        print(f'output: {output}')
        result = torch.argmax(output)
        output_list.append(result.item())

    print(output_list)
    return output_list

if __name__ == '__main__':
    main()