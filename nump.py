import random
from data_loader import load_data

w = random.uniform(-0.1, 0.1)
b = random.uniform(-0.1, 0.1)

data = load_data()


def get_loss(x, y):
    loss = x-y
    return loss*loss


def guess(x: float):
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
    for x, y in data:
        guessed = guess(x)
        evaluate = get_loss(y, guessed)
        losses.append(evaluate)
    print(average_loss(losses))


main()
