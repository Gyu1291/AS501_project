from torch import nn
import torch
from as501_modules import load_hex_file_to_tensor


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
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
        return x
    
    def initializae_parameter(self, weight_dict, bias_dict):
        for name, _ in self._modules.items():
            if name != 'relu':
                self._modules[name].weight = torch.nn.Parameter(weight_dict[name])
                self._modules[name].bias = torch.nn.Parameter(bias_dict[name])

def read_input(file_name):
    tensor = load_hex_file_to_tensor(file_name)
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


def main():
    model = MLP()
    weight_dict, bias_dict = read_parameter()
    print(model._modules)
    model.initializae_parameter(weight_dict, bias_dict)

    batch_input = read_input('image/image_10.txt')
    output_list = []
    
    for single_input in batch_input:
        output = model(single_input)
        result = torch.argmax(output)
        output_list.append(result)
        print(f'output_tensor: {output}, result: {result}')

    print(output_list)
    return output_list
    


if __name__ == '__main__':
    main()