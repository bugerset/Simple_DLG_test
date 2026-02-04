import numpy as np
from data import cifar10, mnist
from data.partition import IID_partition, NIID_partition, print_label_counts
from fl.fedavg import fedavg
from models.simplenet import SimpleCNN
from utils.seed import set_seed
from utils.parser import parse_args
from utils.device import select_device
from utils.plotting import plotting
from attack.noise import make_noise
from attack.generator import generate

def main():
    args = parse_args()
    set_seed(args.seed, True)
    rng = np.random.default_rng(args.seed)

    device = select_device(args.device)
    print(f"Device => {device}")

    if args.data_set == "cifar10":
        train_ds = cifar10.get_cifar10(root=args.data_root)
        global_model = SimpleCNN(in_channel=3).to(device)
    else:
        train_ds = mnist.get_mnist(root=args.data_root)
        global_model = SimpleCNN(in_channel=1).to(device)

    if args.partition == "niid":
        clients = NIID_partition(train_ds, num_clients=args.num_clients, seed=args.seed, alpha=args.alpha, min_size=args.min_size)
    else:
        clients = IID_partition(train_ds, num_clients=args.num_clients, seed=args.seed)

    if args.print_labels:
        print("\n=== Client label distributions ===")
        print_label_counts(train_ds, clients, num_classes=10)

    selected = rng.choice(args.num_clients, size=1, replace=False).item()
    
    # Get client's gradient
    original_data, original_label, client_gradient = fedavg(global_model, clients[selected], batch_size=args.batch_size, lr=args.lr, device=device)

    # Make Attacker's dummy data
    dummy_x, dummy_y, mean, std = make_noise(args.data_set, device, batch_size=args.batch_size)

    recon_x, recon_y = generate(global_model, client_gradient, dummy_x, dummy_y, mean, std, device, grad_amp=args.grad_amp, iter=args.attack_iter)

    plotting(original_data, original_label, recon_x, recon_y, mean, std)

if __name__ == "__main__":
    main()
