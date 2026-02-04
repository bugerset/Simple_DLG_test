# Simple_DLG_test (PyTorch)

A minimal PyTorch implementation of **Deep Leakage from Gradients (DLG)** on **MNIST** and **CIFAR-10**.
Given a client’s shared gradients (typically **batch size = 1**), this code reconstructs the input image and label by **gradient matching**.

> ⚠️ This repo is for **research/educational** purposes only.

---

## Overview

DLG assumes an attacker (e.g., server) can access:
- the **global model parameters** used by the client, and
- the **client gradients** computed from a **single(or more) mini-batch** (batch size = 1 works best).

The attacker then optimizes dummy variables `(x', y')` such that:
`∇θ L(fθ(x'), y') ≈ ∇θ L(fθ(x), y)`.

---

## Threat Model (Assumptions)

- The attacker has access to the **global model architecture and weights** at the round of interest.
- The attacker can observe the **client gradients** from **one mini-batch** (default: **batch size = 1**).
- The attacker does **not** have access to the client’s raw data or labels.

---

## Method (DLG)

We optimize:
- `dummy_x` (image in pixel space, clamped to [0,1])
- `dummy_y` (label logits -> softmax)

by minimizing the gradient matching objective:
`J = Σ ||g_dummy - g_client||²`.

---

## Normalization (Mean / Std)

To match the client-side preprocessing, we use dataset-specific normalization statistics:

- **MNIST**
  - mean = (0.1307,)
  - std  = (0.3081,)

- **CIFAR-10**
  - mean = (0.4914, 0.4822, 0.4465)
  - std  = (0.2470, 0.2435, 0.2616)

During the attack, `dummy_x` is maintained in pixel space **[0, 1]** and normalized **only before** the forward pass:
`x_hat = Normalize(dummy_x, mean, std)`.

Code reference:
- `attack/noise.py`: provides `mean/std` for dummy initialization
- `attack/generator.py`: applies `TF.normalize(dummy_x, mean, std)` before feeding into the model
- `utils/plotting.py`: denormalizes the original input for visualization

---

## Implementation Details (Math ↔ Code)

DLG reconstructs `(x, y)` by solving a **gradient matching** problem.

### Goal (gradient matching)

$$\
\min_{x',y'} \big\|\nabla_{\theta} L(f_{\theta}(x'),y') - g_{\text{client}}\big\|_2^2
\$$

**Intuition:** we update `dummy_x` (and `dummy_y`) to **minimize the gradient difference** between the client’s gradient and the dummy gradient.  
In other words, `dummy_x` is optimized using **J**, the discrepancy between `g_dummy` and `g_client`, so that the dummy sample produces gradients indistinguishable from the client’s.

### Math ↔ Code mapping

- **Client gradient extraction**: `fl/fedavg.py`  
  - returns `{name: grad}` for each parameter (batch size = `--batch-size`)

- **Dummy initialization**: `attack/noise.py`  
  - returns `dummy_x ∼ U(0,1)` (pixel space), `dummy_y ∼ N(0,1)` (label logits)

- **DLG optimization loop**: `attack/generator.py`  
  - **LBFGS** optimizes `[dummy_x, dummy_y]` *(model weights are fixed)*  
  - `dummy_y → softmax(dummy_y)` yields differentiable soft labels  
  - `TF.normalize(dummy_x, mean, std)` matches the client-side preprocessing  
  - `J = Σ ||g_dummy - g_client||²` is computed and backpropagated to update `dummy_x, dummy_y`
Key code (simplified):

```python
# attack/generator.py
optimizer = torch.optim.LBFGS([dummy_x, dummy_y], lr=1, max_iter=20, history_size=100, line_search_fn='strong_wolfe')

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
```

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

---

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

---

## CLI Arguments

Key arguments (from utils/parser.py):
```
	•	Reproducibility / compute
		•	--seed (default: 845)
		•	--device in {auto,cpu,cuda,mps}

	•	Training parameter
		•	--grad-amp (default 1e2)

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

---

## Expected Output

(*wit



