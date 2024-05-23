import torch

tensor1 = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
tensor2 = torch.tensor([[1,2],
                        [3,4],
                        [5,6]])

print(torch.matmul(tensor1, tensor2))