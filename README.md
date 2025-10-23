# Deep Learning From Scratch

Building my own implementations of deep learning primitives to truly understand how they work under the hood.

## Philosophy

This repo contains multiple versions (`v1/`, `v2/`, etc.) as I iterate and learn. Each version represents a deeper understanding or different approach to building ML frameworks from first principles.

---

## v1 - Scalar Autograd & Linear Regression ✅

A **complete** minimal automatic differentiation engine using scalar values.

**What's implemented:**
- Custom autograd engine with computational graph tracking
- Reverse-mode automatic differentiation (backpropagation via topological sort)
- Basic operations: add, subtract, multiply, divide, power
- SGD optimizer with gradient zeroing
- Synthetic data generation (y = x - 1 relationship)
- Linear regression training loop (50 epochs)

**Key concepts demonstrated:**
- Computational graphs as DAGs
- Chain rule application through backward pass
- Gradient accumulation
- Operator overloading for clean syntax

**Generate data:**
```bash
python -m v1.create_data
```

**Train model:**
```bash
python -m v1.train
```

**Deep dive:** See `v1/README.md` for comprehensive explanation of every concept.

---

## v2 - Tensor Autograd (🚧 In Progress)

Upgrading from scalar to **tensor operations** using NumPy arrays.

**What's changing:**
- `value` is now `np.array()` instead of Python scalar
- `grad` is now `np.zeros_like()` for matching shapes
- Operations handle element-wise and matrix computations
- Real-world dataset: Kaggle gender classification (Gender, Age, Height → Income)

**Current status:**
- ⚠️ `autograd.py` has syntax error on line 32 in `multiply()` 
- ⚠️ `train.py` is exploration code only (incomplete training loop)
- ⚠️ `optimizer.py` needs update for NumPy arrays
- ✅ Data loader working with normalized features

**Dataset:**
- Source: `muhammadtalharasool/simple-gender-classification`
- Features: Gender (binary), Age, Height (cm) 
- Target: Income (USD)
- All normalized to [0, 1] range

**Next steps:**
1. Fix multiply operation syntax
2. Implement proper matrix multiplication
3. Build complete training loop
4. Add batch processing

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
├── v2/              # Tensor autograd (WIP)
│   ├── autograd.py      # NumPy-based engine (has bugs)
│   ├── optimizer.py     # Same as v1 (needs update)
│   ├── data_loader.py   # Kaggle dataset loader
│   └── train.py         # Exploration code (incomplete)
│
├── pyproject.toml   # Dependencies (numpy, pandas, kagglehub, torch)
└── README.md        # This file
```

---

## Dependencies

Install with `uv`:
```bash
uv sync
```

Or with pip:
```bash
pip install numpy pandas matplotlib kagglehub torch
```

---

## Learning Goals

By building these frameworks from scratch, I'm learning:

1. **How autograd actually works** - not just using PyTorch/TensorFlow
2. **Computational graphs** - how every operation becomes a node
3. **Backpropagation** - chain rule + topological sort
4. **Gradient accumulation** - why we zero gradients
5. **Operator overloading** - making Python do what we want
6. **Scalar → Tensor transition** - handling shape broadcasting

---

## What Makes This Different

This isn't about building a "better PyTorch" - it's about **understanding** PyTorch by reimplementing its core ideas. Every line of code here teaches something fundamental about how modern ML frameworks work under the hood.

The goal: when I use `loss.backward()` in PyTorch, I **know** exactly what's happening, not just that "it works."

---

*This is learning by building, not production code. Bugs are features (learning opportunities).*

