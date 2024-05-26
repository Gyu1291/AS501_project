import model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

i = 2
my_model = model.MLP()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=0.01)

train_dataset = dataset.MNIST(root='../dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dataset.MNIST(root='../dataset/', train=False, transform=transforms.PILToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# TensorBoard writer
writer = SummaryWriter()

my_model = torch.load('my_model.pt')
my_model.to(device)

# for epoch in range(i):
#     count = 0
#     train_correct = 0
#     train_total = 0
#     epoch_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         data = data.view(-1, 784)
#         data = data.type(torch.float32)
#         output = my_model(data)
        
#         loss = criterion(output, target)
#         epoch_loss += loss.item()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         _, predicted = torch.max(output.data, 1)
#         train_total += target.size(0)
#         train_correct += (predicted == target).sum().item()

#         if batch_idx % 100 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')

#     # Calculate accuracy

#     train_accuracy = 100. * train_correct / train_total
#     epoch_loss /= len(train_loader)

#     # Log to TensorBoard
#     writer.add_scalar('Loss/train', epoch_loss, epoch)
#     writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    
#     print(f'Train Accuracy: {train_accuracy:.2f}%')

# torch.save(my_model, 'my_model.pt')
# Close the writer
my_model.eval()
test_correct = 0
test_total = 0

for batch_idx, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, 784)
    data = data.type(torch.float32)
    output = my_model(data)
    print(target,output)
    
    loss = criterion(output, target)

    _, predicted = torch.max(output.data, 1)
    test_total += target.size(0)
    test_correct += (predicted == target).sum().item()

    if batch_idx % 100 == 0:
        print(f'[{batch_idx * len(data)}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader)}%)]\tLoss: {loss.item()}')
test_accuracy = 100. * test_correct / test_total

print(f'Accuracy: {test_accuracy:.2f}%')
writer.close()
