from torchvision import datasets, transforms

def get_mnist(root="./data"):

    # Normalize MNIST with below values
    mean = (0.1307,)
    std  = (0.3081,)

    train_tf = []

    train_tf.append(transforms.ToTensor())
    
    train_tf.append(transforms.Normalize(mean, std))


    train_data = datasets.MNIST(
        root=root,   
        train=True, 
        download=True,
        transform=transforms.Compose(train_tf)
    )

    return train_data