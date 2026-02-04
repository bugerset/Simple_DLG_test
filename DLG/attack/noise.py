import torch

def make_noise(dataset, device, batch_size):
    if dataset == "cifar10":
        shape_x = (batch_size, 3, 32, 32)
        num_classes = 10
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    else:
        shape_x = (batch_size, 1, 28, 28)
        num_classes = 10
        mean = torch.tensor([0.1307], device=device).view(1, 1, 1, 1)
        std = torch.tensor([0.3081], device=device).view(1, 1, 1, 1)

    dummy_x = torch.randn(shape_x, device=device).requires_grad_(True)
    
    dummy_y = torch.randn((batch_size, num_classes), device=device).requires_grad_(True)

    return dummy_x, dummy_y, mean, std