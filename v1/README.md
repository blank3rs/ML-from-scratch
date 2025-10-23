# v1 - Autograd & Linear Regression: A Deep Dive

A minimal automatic differentiation engine and training system built from first principles. This document explains **everything** I learned building this.

## What This Is

This is **v1** of a from-scratch deep learning framework. It implements:

- Custom autograd engine with computational graph
- Reverse-mode automatic differentiation (backpropagation)
- SGD optimizer
- Linear regression training loop

The goal: understand how modern ML frameworks work under the hood.

---

## The Big Picture: Computational Graphs

### What Are They?

Every mathematical expression can be represented as a directed acyclic graph (DAG):

```
                w = 0.5
                    ↓
                [MULTIPLY] ← x = 2
                    ↓
                result1 = 1.0
                    ↓
                [ADD] ← b = 1
                    ↓
              y_hat = 2.0
                    ↓
                [SUBTRACT] ← y = 3
                    ↓
                diff = -1.0
                    ↓
                [POWER(2)]
                    ↓
                loss = 1.0
```

### Why Graphs Matter

**Forward Pass**: Compute values flowing DOWN the graph  
**Backward Pass**: Compute gradients flowing UP the graph

This structure lets us efficiently compute ALL derivatives of a loss with respect to ALL inputs in a single backward pass.

---

## Core Design: The `autograd` Class

### Node Structure

Every node in the computational graph is an `autograd` object:

```python
class autograd:
    def __init__(self, value, parents=(), op=''):
        self.value = value      # The computed value
        self.parents = parents  # Nodes that produced this value
        self.op = op            # Operation name (debugging)
        self.grad = 0.0         # Gradient (initially zero)
        self._backward = lambda: None  # Function to compute gradients
```

**Key insight**: Each node stores its own gradient computation function! This is how we get automatic differentiation.

### How Operations Work

Let's trace through `multiply(a, b)`:

```python
@staticmethod
def multiply(a, b):
    # 1. Create output node with computed value
    out = autograd(a.value * b.value, (a, b), '*')
    
    # 2. Define gradient computation using chain rule
    out._backward = lambda: (
        setattr(a, 'grad', a.grad + b.value * out.grad),
        setattr(b, 'grad', b.grad + a.value * out.grad)
    )
    
    return out
```

**What's happening**:
- Forward: `out.value = a.value * b.value`
- Backward: When we have `out.grad`, update `a.grad` and `b.grad`

**The math**: For `z = x * y`:
- `∂z/∂x = y`
- `∂z/∂y = x`

So with chain rule:
- `∂loss/∂x = (∂loss/∂z) * (∂z/∂x) = out.grad * y`
- `∂loss/∂y = (∂loss/∂z) * (∂z/∂y) = out.grad * x`

**Accumulation**: Note `a.grad +` - we ADD because gradients from multiple paths accumulate.

### All Operations Explained

#### Addition: `add(a, b)`
```python
out._backward = lambda: (
    setattr(a, 'grad', a.grad + 1.0 * out.grad),
    setattr(b, 'grad', b.grad + 1.0 * out.grad)
)
```
**Math**: `∂(x+y)/∂x = 1`, `∂(x+y)/∂y = 1`  
Both inputs get the gradient equally.

#### Subtraction: `sub(a, b)`
```python
out._backward = lambda: (
    setattr(a, 'grad', a.grad + 1.0 * out.grad),
    setattr(b, 'grad', b.grad - 1.0 * out.grad)  # NEGATIVE!
)
```
**Math**: `∂(x-y)/∂x = 1`, `∂(x-y)/∂y = -1`  
Second input gets NEGATIVE gradient.

#### Division: `div(a, b)`
```python
out._backward = lambda: (
    setattr(a, 'grad', a.grad + (1 / b.value) * out.grad),
    setattr(b, 'grad', b.grad - (a.value / (b.value ** 2)) * out.grad)
)
```
**Math**: For `z = x/y`:
- `∂z/∂x = 1/y`
- `∂z/∂y = -x/y²`

#### Power: `pow(a, exp)`
```python
out._backward = lambda: (
    setattr(a, 'grad', a.grad + exp * (a.value ** (exp - 1)) * out.grad)
)
```
**Math**: For `z = x^exp`:
- `∂z/∂x = exp * x^(exp-1)`

**Example**: If `z = x²`, then `∂z/∂x = 2x` ✓

---

## The Magic: Backward Pass

### Why Topological Sort?

**Critical insight**: Must process nodes in REVERSE order of dependencies.

Can't compute gradients for a node before computing gradients for its children!

### The Algorithm

```python
def backward(self):
    topo, visited = [], set()
    
    def build(v):
        if v not in visited:
            visited.add(v)
            for p in v.parents:  # Visit parents first
                build(p)
            topo.append(v)  # Add after all dependencies
    
    build(self)
    self.grad = 1.0  # Initialize output gradient
    
    for node in reversed(topo):  # Process in reverse order
        node._backward()
```

**Step-by-step example**:

For graph: `w → [*] → result → [+] → y_hat → [-] → loss`

1. **Build topological order**:
   - DFS visits: w, b, x, y
   - Adds: w, b, x, y
   - Then: [*] node
   - Then: [+] node
   - Then: y_hat, loss

2. **Reverse order**: `[loss, y_hat, [+], [*], w, b, x, y]`

3. **Initialize**: `loss.grad = 1.0`

4. **Process backwards**:
   - `loss._backward()` → updates `y_hat.grad`
   - `y_hat._backward()` → updates `[+].grad`
   - `[+]._backward()` → updates `result.grad` and `b.grad`
   - `[*]._backward()` → updates `w.grad` and `x.grad`

### Why This Works

**Chain rule**: If `loss = f(g(h(x)))`:
```
∂loss/∂x = (∂loss/∂f) * (∂f/∂g) * (∂g/∂h) * (∂h/∂x)
```

Each node multiplies its local derivative by the incoming gradient.

**Visual**:
```
loss.grad = 1.0
    ↓
y_hat._backward(): diff.grad = 2 * loss.grad = 2.0
    ↓
diff._backward(): y_hat.grad = 1.0 * diff.grad = 2.0
    ↓
y_hat._backward(): result.grad = 1.0 * y_hat.grad = 2.0
                    b.grad = 1.0 * y_hat.grad = 2.0
    ↓
result._backward(): w.grad = x.value * result.grad = 4.0
                    x.grad = w.value * result.grad = 1.0
```

---

## Training Loop: How It All Fits Together

### The Full Flow

```python
# 1. Initialize parameters
w = autograd(random.uniform(-0.1, 0.1), ())
b = autograd(random.uniform(-0.1, 0.1), ())

# 2. For each training example
for x, y in data:
    # Forward pass
    x = autograd(x, ())
    y_hat = w * x + b  # Creates computational graph
    
    # Compute loss
    y = autograd(y, ())
    loss = (y - y_hat) ** 2
    
    # Backward pass
    loss.backward()  # Computes gradients for w and b
    
    # Update parameters
    opt.adjust(w)  # w.value -= learning_rate * w.grad
    opt.adjust(b)  # b.value -= learning_rate * b.grad
```

### What Happens at Each Step

**Forward**:
```
x (0.5) → [MULTIPLY] with w → result (0.025)
                                ↓
                         [ADD] with b → y_hat (0.045)
                                ↓
                         [SUBTRACT] y → diff (-0.05)
                                ↓
                         [POWER(2)] → loss (0.0025)
```

**Backward**:
```
loss.grad = 1.0
    ↓
diff.grad = 2 * diff.value * loss.grad = -0.1
    ↓
y_hat.grad = 1.0 * diff.grad = -0.1
    ↓
result.grad = 1.0 * y_hat.grad = -0.1
    ↓
w.grad = x.value * result.grad = -0.05
    ↓
Update: w.value -= 0.00001 * (-0.05) = +0.0000005
```

Gradient is negative → loss decreases as w increases → update increases w ✓

---

## Optimizer: Gradient Descent

### Simple SGD

```python
class optimizer:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
    
    def adjust(self, param):
        param.value = param.value - (self.learning_rate * param.grad)
        param.grad = 0  # Reset for next iteration
```

**What it does**:
- Moves parameter opposite to gradient direction
- Learning rate controls step size
- Zeros gradient after update

**Why reset gradients**:
- Each forward pass accumulates gradients
- Must start fresh for next iteration

---

## Data Pipeline

### Generation (`create_data.py`)

Creates relationship: `y = x - 1`

```python
for i in range(100000):
    file.write(f"{i}:{i-1}\n")
```

### Loading (`data_loader.py`)

```python
def load_data():
    export_data = []
    with open("data.txt", 'r') as file:
        for line in file:
            x, y = line.split(":")
            export_data.append([float(x)/100000, float(y)/100000])
    return np.array(export_data)
```

**Normalization**: Divide by 100,000 to keep values in [0, 1]. Prevents numerical overflow.

---

## Model: Linear Regression

### The Model

```python
y_hat = w * x + b
```

**Goal**: Learn `w` and `b` to minimize `(y - y_hat)²`

### Expected Outcome

Training on `y = x - 1` (normalized), model should learn:
- `w ≈ 1.0` (slope)
- `b ≈ -1.0` (offset)

### Loss Function

Mean Squared Error:
```python
loss = (y - y_hat) ** 2
```

Properties:
- Always positive
- Squared amplifies large errors
- Smooth and differentiable

---

## Key Learnings

### 1. Computational Graphs Are Everywhere

Every ML framework uses this same structure:
- PyTorch: `torch.Tensor` nodes
- TensorFlow: Operations in `tf.Graph`
- JAX: Pure functions with automatic differentiation

### 2. Backpropagation Is Just Chain Rule

Nothing magical - just systematic application of calculus.

### 3. Forward vs Backward Pass Are Separate

- Forward: Compute values
- Backward: Compute gradients
- Can't mix them!

### 4. Gradients Accumulate

If multiple paths lead to a parameter, gradients sum:
```
loss → path1 → w
loss → path2 → w
w.grad = grad1 + grad2
```

### 5. Topological Sort Is Critical

Wrong order = wrong gradients. Must process children before parents.

### 6. Autograd Enables Clean Code

Operator overloading (`__add__`, `__mul__`) lets you write:
```python
loss = (w * x + b - y) ** 2
```

Instead of manually computing gradients!

---

## Usage

**Generate data**:
```bash
python -m v1.create_data
```

**Train model**:
```bash
python -m v1.train
```

Expected convergence over 50 epochs.

---

## Limitations (V1)

By design, this is minimal:

- Single scalar values only (no tensors/arrays)
- Manual parameter collection
- No batching (sequential processing)
- Simple SGD only (no momentum, Adam, etc.)
- No computational optimizations

**Why**: Focus on understanding the core algorithm first.

---

## What Comes Next

Future versions might add:
- **v2**: Tensors/arrays instead of scalars
- **v3**: Neural networks with multiple layers
- **v4**: Batch processing
- **v5**: Advanced optimizers (Adam, etc.)
- **v6**: GPU acceleration

---

## Why This Matters

You're implementing the **same core algorithm** that powers:
- PyTorch
- TensorFlow  
- JAX
- Any modern deep learning framework

Understanding this makes you **know** how ML training actually works, not just how to use libraries. This is foundational knowledge for becoming a better ML engineer.
