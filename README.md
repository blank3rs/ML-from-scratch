# Deep Learning From Scratch

Building my own implementations of deep learning primitives to truly understand how they work under the hood.

## Philosophy

This repo contains multiple versions (`v1/`, `v2/`, etc.) as I iterate and learn. Each version represents a deeper understanding or different approach to building ML frameworks from first principles.

## v1 - Autograd & Linear Regression

A minimal automatic differentiation engine with basic optimization.

**What's included:**
- Custom autograd engine with computational graph and backpropagation
- SGD optimizer
- Linear regression training loop
- Basic data loading

**Run it:**
```bash
python train.py
```

**Test gradients:**
```bash
python -m v1.test_autograd
```

---

This is about learning by building, not creating production libraries. The goal is understanding.

