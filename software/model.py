from torch import nn
import torch
from as501_modules import load_hex_file_to_tensor, hex_to_int, load_input_to_utensor
from analyze_tensor import plot_heatmap, summarize_tensor, calculate_sparsity, uniform_quantization

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x
    
    def initializae_parameter(self, weight_dict, bias_dict):
        for name, _ in self._modules.items():
            if name != 'relu':
                self._modules[name].weight = nn.Parameter(uniform_quantization(weight_dict[name], 4))
                self._modules[name].bias = nn.Parameter(uniform_quantization(bias_dict[name], 4))


def read_input(file_name):
    tensor = load_input_to_utensor(file_name)
    # Reshape the tensor to (1, 784)
    tensor = tensor.reshape(-1, 784)
    return tensor

def read_parameter():
    weight_dict = {}
    bias_dict = {}
    weight_dict['fc1'] = load_hex_file_to_tensor('parameter/weight1.txt')
    weight_dict['fc2'] = load_hex_file_to_tensor('parameter/weight2.txt')
    weight_dict['fc3'] = load_hex_file_to_tensor('parameter/weight3.txt')

    bias_dict['fc1'] = load_hex_file_to_tensor('parameter/bias1.txt')
    bias_dict['fc2'] = load_hex_file_to_tensor('parameter/bias2.txt')
    bias_dict['fc3'] = load_hex_file_to_tensor('parameter/bias3.txt')
    
    return weight_dict, bias_dict

def check_accuracy(output_list, label_list):
    correct = 0
    for i in range(len(output_list)):
        if output_list[i] == label_list[i]:
            correct += 1
    return correct


def main():

    model = MLP()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    weight_dict, bias_dict = read_parameter()
    model.initializae_parameter(weight_dict, bias_dict)

    batch_input = read_input('image/image_10000.txt')
    output_list = []

    batch_input = batch_input.to(device)
    model.to(device)

    i=0
    for single_input in batch_input:
        output = model(single_input)

        result = torch.argmax(output)
        if device != 'cpu':
            result = result.to('cpu')
        output_list.append(result.item())
        i+=1
    label_file_path = 'label/label.txt'
    with open(label_file_path, 'r') as f:
        labels = []
        for _ in range(len(output_list)):
            line = f.readline()
            if not line:
                break
            labels.append(line)
    temp = [line.strip() for line in labels]
    label_list = [hex_to_int(line) for line in temp]

    # print(output_list)
    # print(label_list)
    correctness = check_accuracy(output_list, label_list)

    print(f'correctness: {correctness}, total: {len(output_list)} accuracy: {(correctness / len(output_list))*100}%')
    print(f'Average input sparsity: {calculate_sparsity(batch_input)*100}%')
    return output_list

if __name__ == '__main__':
    main()