import random
from v1 import load_data, autograd

w = autograd(random.uniform(-0.1, 0.1), ())
b = autograd(random.uniform(-0.1, 0.1), ())

data = load_data()


def get_loss(y, y_hat):
    y = autograd(y, ())
    loss = y-y_hat
    loss = loss**2
    loss.backward()
    print("loss:", loss.value)
    print("w.grad:", w.grad)
    print("b.grad:", b.grad)
    b.adjust()
    w.adjust()

    return loss.value


def guess(x: float):
    x = autograd(x, ())
    y_hat = w*x+b

    return y_hat


def average_loss(arr):
    sum = 0
    count = 0
    for i in arr:
        sum += i
        count += 1
    return sum/count


def main():
    losses = []
    for i in range(50):
        for x, y in data:
            guessed = guess(x)
            evaluate = get_loss(y, guessed)
            losses.append(evaluate)
        print(average_loss(losses))

    test1 = guess(2)
    test2 = guess(26)
    test3 = guess(612)
    test4 = guess(150)

    print(test1)
    print(test2)
    print(test3)
    print(test4)


main()
