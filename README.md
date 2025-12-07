# ML From Scratch

Implementations of machine learning primitives from scratch.

This repo contains multiple versions as I iterate and learn. Each version represents a different approach to building ML frameworks.

---

## v1 - Scalar Autograd

Automatic differentiation engine using scalar values.

**What's implemented:**
- Autograd engine with computational graph tracking
- Reverse-mode automatic differentiation
- Operations: add, subtract, multiply, divide, power
- SGD optimizer
- Synthetic data generation
- Linear regression training

**Generate data:**
```bash
uv run python -m v1.create_data
```

**Train model:**
```bash
uv run python -m v1.train
```

**Deep dive:** See `v1/README.md` for comprehensive explanation of every concept.

---

## v2 - Tensor Autograd

Automatic differentiation engine using NumPy arrays for matrix operations.

**What's implemented:**
- Autograd engine supporting NumPy arrays
- Element-wise operations: add, subtract, multiply, divide, power
- Matrix multiplication with gradient propagation
- SGD optimizer for tensor parameters
- Train/test split and evaluation metrics
- Real dataset: Kaggle gender classification

**Dataset:**
- Source: `muhammadtalharasool/simple-gender-classification`
- Features: Gender, Age, Height
- Target: Income
- Normalized to [0, 1] range

**Train model:**
```bash
uv run v2/train.py
```

---

## Project Structure

```
learn/
├── v1/              # Scalar autograd (complete)
│   ├── autograd.py      # Core engine
│   ├── optimizer.py     # SGD implementation
│   ├── data_loader.py   # Load synthetic data
│   ├── create_data.py   # Generate training data
│   ├── train.py         # Training loop
│   └── README.md        # Deep technical explanation
│
├── v2/              # Tensor autograd
│   ├── autograd.py      # NumPy-based engine
│   ├── optimizer.py     # SGD optimizer
│   ├── data_loader.py   # Dataset loader with train/test split
│   └── train.py         # Training loop with evaluation
│
├── pyproject.toml   # Dependencies (numpy, pandas, kagglehub, torch)
└── README.md        # This file
```

---

## Setup

This project uses `uv` for dependency management.

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install dependencies:**
```bash
uv sync
```

**Run with uv:**
```bash
uv run python -m v1.create_data
uv run python -m v1.train
uv run v2/train.py
```

---

## Learning Goals

Building these frameworks from scratch to understand:

1. How autograd works internally
2. Computational graphs and node tracking
3. Backpropagation via topological sort
4. Gradient accumulation and zeroing
5. Operator overloading for clean syntax
6. Shape handling and broadcasting with tensors

