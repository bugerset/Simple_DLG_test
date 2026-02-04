import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def generate(global_model, c_grads, dummy_x, dummy_y, mean, std, device, grad_amp=1e2, iter=500):
    global_model.to(device)
    global_model.eval()

    c_grads = {k: v.to(device) for k, v in c_grads.items()}

    mean_c = mean.view(-1).tolist() if torch.is_tensor(mean) else mean
    std_c  = std.view(-1).tolist()  if torch.is_tensor(std)  else std

    optimizer = torch.optim.LBFGS([dummy_x, dummy_y], lr=1, max_iter=20, history_size=100, line_search_fn='strong_wolfe')

    for i in range(1, iter+1):
        def closure():
            with torch.no_grad():
                dummy_x.clamp_(0, 1)

            optimizer.zero_grad()

            x_hat = TF.normalize(dummy_x, mean_c, std_c)

            dummy_pred = global_model(x_hat)
            
            dummy_loss = F.cross_entropy(dummy_pred, torch.softmax(dummy_y, dim=-1))

            dummy_grads_tuple = torch.autograd.grad(dummy_loss, global_model.parameters(), create_graph=True)

            dummy_grads = {name: g for (name, _), g in zip(global_model.named_parameters(), dummy_grads_tuple)}

            grad_diff = 0

            for name in c_grads.keys():
                diff = (dummy_grads[name] - c_grads[name]).pow(2).sum()
                grad_diff += diff

            grad_diff *= grad_amp
            
            grad_diff.backward()

            return grad_diff
        
        loss_val = optimizer.step(closure)

        if i % 10 == 0:
            print(f"[Iter{i}] Loss => {loss_val.item():.20f}")

    return dummy_x.detach().cpu(), dummy_y.detach().cpu()