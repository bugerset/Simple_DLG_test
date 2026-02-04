import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def fedavg(global_model, client_dataset, batch_size=1, lr=1e-3, device="cpu"):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.eval()
    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = local_model(x)
        loss = criterion(pred, y)
        loss.backward()

        # Leakage of Gradient
        grads = {}
        for name, p in local_model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().clone().cpu()

        return x, y, grads