import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def plotting(original_x, original_y, recon_x, recon_y, mean, std):
    batch_size = original_x.shape[0]
    
    fig, axes = plt.subplots(batch_size, 2, figsize=(14,8), squeeze=False)

    for i in range(batch_size):
        original_img = original_x[i].detach().cpu()
        true_label = original_y[i].item()

        m = mean[0].cpu() if mean.dim() > 1 else mean.cpu()
        s = std[0].cpu() if std.dim() > 1 else std.cpu()

        original_img = original_img * s + m

        original_img = torch.clamp(original_img, 0, 1)

        if original_img.shape[0] == 3:
            axes[i,0].imshow(to_pil_image(original_img))
        else:
            axes[i,0].imshow(to_pil_image(original_img), cmap="gray")
        
        axes[i,0].set_title("Original (label={})".format(true_label))
        axes[i,0].axis('off')

        recon_img = recon_x[i].detach().cpu()

        recon_img = torch.clamp(recon_img, 0, 1)
        
        pred_label = torch.argmax(recon_y[i], dim=-1).item()
        
        if recon_img.shape[0] == 3:
            axes[i,1].imshow(to_pil_image(recon_img))
        else:
            axes[i, 1].imshow(to_pil_image(recon_img), cmap="gray")
            
        axes[i,1].set_title(f"Reconstructed {i} (Label: {pred_label})")
        axes[i,1].axis('off')

    plt.tight_layout()
    plt.show()