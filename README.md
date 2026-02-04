# Simple_DLG_test (PyTorch)

A minimal PyTorch implementation of **Deep Leakage from Gradients (DLG)** on **MNIST** and **CIFAR-10**.
Given a client’s shared gradients (typically **batch size = 1**), this code reconstructs the input image and label by **gradient matching**.

> ⚠️ This repo is for **research/educational** purposes only.


## Overview

DLG assumes an attacker (e.g., server) can access:
- the **global model parameters** used by the client, and
- the **client gradients** computed from a **single(or more) mini-batch** (batch size = 1 works best).

The attacker then optimizes dummy variables `(x', y')` such that:
`∇θ L(fθ(x'), y') ≈ ∇θ L(fθ(x), y)`.


## Threat Model (Assumptions)

- The attacker has access to the **global model architecture and weights** at the round of interest.
- The attacker can observe the **client gradients** from **one mini-batch** (default: **batch size = 1**).
- The attacker does **not** have access to the client’s raw data or labels.


## Method (DLG)

We optimize:
- `dummy_x` (image in pixel space, clamped to [0,1])
- `dummy_y` (label logits -> softmax)

by minimizing the gradient matching objective:
`Gradient_Distance = Σ ||g_dummy - g_client||²`.


## Recommended Folder Structure

Your `main.py` imports modules like `data.cifar10`, `fl.fedavg`, etc.  
So the easiest way to run without changing code is to organize files like this:
```
├── main.py
├── attack/
│   ├── __init__.py
│   ├── generator.py
│   └──  noise.py
├── data/
│   ├── __init__.py
│   ├── cifar10.py
│	├── mnist.py
│   └── partition.py
├── fl/
│   ├── __init__.py
│   └── fedavg.py
├── models/
│   ├── __init__.py
│   └── simplenet.py
└── utils/
 	├── __init__.py
	├── plotting.py
 	├── device.py
    ├── eval.py
    ├── parser.py
    └── seed.py
```

---

## Requirements

- Python 3.9+ recommended
- PyTorch + torchvision
- numpy

Run with default settings:
```bash
python main.py
```
Examples: 
```bash
python main.py --data-set mnist --attack-iter 200
python main.py --data-set cifar10 --attack-iter 500
python main.py --data-set mnist --attack-iter 100 --grad-amp 1e4 --batch-size 8
```

## Device Selection

The code supports:
```
	•	--device auto (default): selects CUDA if available, else MPS (Apple Silicon), else CPU
	•	--device cuda
	•	--device mps
	•	--device cpu
```

Example:
```bash
python main.py --device auto
```

## CLI Arguments

Key arguments (from utils/parser.py):
```
	•	Reproducibility / compute
		•	--seed (default: 845)
		•	--device in {auto,cpu,cuda,mps}

	•	Train parameter
		•	--dyn-alpha (FedDyn alpha, default 0.1)

	•	Dataset
		•   --data-set (default cifar10, choices=[cifar10, mnist])
		•	--data-root (default ./data)

	•	Fed learning & Attack iter
		•	--num-clients (default 10)
		•	--client-frac fraction of clients sampled per round (default 0.25)
		•	--batch-size (default 1)
		•	--attack-iter (default=100)

	•	Data partitioning
		•	--partition in {iid,niid}
		•	--alpha: Dirichlet concentration parameter controlling Non-IID severity.
		    	├── α = 0.1 ~ 0.3: highly skewed label distribution (strong Non-IID)
		  		├──	α = 0.5: moderate Non-IID (default)
		  		└──	α = 0.8 ~ 1.0: closer to IID
		•	--min-size minimum samples per client in non-IID (default 10)
		•	--print-labels / --no-print-labels
```

## Expected Output

(*wit



