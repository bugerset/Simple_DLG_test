import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Federate Learning MobileNet CIFAR-10 and MNIST")

    # seed, device, train function setting
    parser.add_argument("--seed", type=int, default=845)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    # Training parameter
    parser.add_argument("--grad-amp", type=float, default=1e2)
    parser.add_argument("--lr", type=float, default=1)

    # Dataset setting
    parser.add_argument("--data-set", type=str, default="mnist, choices=["cifar10", "mnist"])
    parser.add_argument("--data-root", type=str, default="./data")

    # Client, Batch, Local Epochs, Communicate rounds setting
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--client-frac", type=float, default=0.2)
    parser.add_argument("--attack-iter", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)

    # IID, N-IID setting
    parser.add_argument("--partition", type=str, default="niid", choices=["iid", "niid"])
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--print-labels", dest="print_labels", action="store_true", default=True)
    parser.add_argument("--no-print-labels", dest="print_labels", action="store_false")

    return parser.parse_args()
