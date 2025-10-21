from autograd import autograd

w = autograd(0.072)
x = autograd(2)
b = autograd(-0.061)
y = autograd(4)

loss = ((w * x + b) - y) ** 2
print("loss:", loss.value)
loss.backward()
print("w.grad:", w.grad)
print("b.grad:", b.grad)
print("x.grad:", x.grad)
print("y.grad:", y.grad)
