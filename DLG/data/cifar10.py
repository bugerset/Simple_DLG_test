from torchvision import datasets, transforms

def get_cifar10(root="./data"):

    # Normalize CIFAR-10 with below values
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = []

    train_tf.append(transforms.ToTensor())
    train_tf.append(transforms.Normalize(mean, std))

    train_data = datasets.CIFAR10(
        root=root,   
        train=True, 
        download=True,
        transform=transforms.Compose(train_tf)
    )

    return train_data