from torch import nn
import torch
from as501_modules import load_hex_file_to_tensor, hex_to_int
from analyze_tensor import plot_heatmap, summarize_tensor, calculate_sparsity

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.fc1_sparsity = []
        self.fc2_sparsity = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        self.fc1_sparsity.append(calculate_sparsity(x)*100)
        x = self.fc2(x)
        x = self.relu(x)
        self.fc2_sparsity.append(calculate_sparsity(x)*100)
        x = self.fc3(x)
        return x
    
    def initializae_parameter(self, weight_dict, bias_dict):
        for name, _ in self._modules.items():
            if name != 'relu':
                self._modules[name].weight = nn.Parameter(weight_dict[name].transpose(0, 1))
                self._modules[name].bias = nn.Parameter(bias_dict[name])

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
    # print(model._modules)
    model.initializae_parameter(weight_dict, bias_dict)
    # print(model._modules['fc1'].weight)
    # print(len(model._modules['fc1'].weight[0]))

    batch_input = read_input('image/image_100.txt')
    output_list = []

    batch_input = batch_input.to(device)
    model.to(device)

    #check image
    #plot_heatmap(batch_input[0].to('cpu').reshape(28, 28), 'sample_image_heatmap')

    for single_input in batch_input:
        output = model(single_input)

        # relu_f = nn.ReLU()
        # x = torch.matmul(single_input, weight_dict['fc1'].to(device)) + bias_dict['fc1'].to(device)
        # x = relu_f(x)
        # x = torch.matmul(x, weight_dict['fc2'].to(device)) + bias_dict['fc2'].to(device)
        # x = relu_f(x)
        # output = torch.matmul(x, weight_dict['fc3'].to(device)) + bias_dict['fc3'].to(device)    

        result = torch.argmax(output)
        if device != 'cpu':
            result = result.to('cpu')
        output_list.append(result.item())

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
    print(f'Average fc1 activation sparsity: {sum(model.fc1_sparsity)/len(model.fc1_sparsity)}%')
    print(f'Average fc2 activation sparsity: {sum(model.fc2_sparsity)/len(model.fc2_sparsity)}%')
    return output_list

if __name__ == '__main__':
    main()