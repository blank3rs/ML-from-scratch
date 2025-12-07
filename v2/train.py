from data_loader import get_train_test_split
from autograd import autograd
from optimizer import optimizer
import numpy as np

x_train, x_test, y_train, y_test = get_train_test_split(test_size=0.2)

weights = autograd(np.random.rand(3, 1) * 0.1)
bias = autograd(np.random.rand(1) * 0.1)

opt = optimizer(learning_rate=0.01)


def forward(x):
    x = autograd(x)
    y_hat = x @ weights + bias
    return y_hat


def compute_loss(y_true, y_pred):
    y_true = autograd(y_true)
    loss = (y_true - y_pred) ** 2
    return loss


def train_epoch():
    total_loss = 0
    count = 0

    for x, y in zip(x_train, y_train):
        x = x.reshape(1, -1)
        y = np.array([y])

        y_pred = forward(x)
        loss = compute_loss(y, y_pred)

        loss.backward()
        opt.adjust(weights)
        opt.adjust(bias)

        total_loss += loss.value.item()
        count += 1

    return total_loss / count


def evaluate(x_data, y_data):
    total_loss = 0
    predictions = []
    actuals = []
    count = 0

    for x, y in zip(x_data, y_data):
        x = x.reshape(1, -1)
        y = np.array([y])

        y_pred = forward(x)
        loss = compute_loss(y, y_pred)

        total_loss += loss.value.item()
        predictions.append(y_pred.value.item())
        actuals.append(y.item())
        count += 1

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    avg_loss = total_loss / count
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

    return avg_loss, mae, rmse, predictions, actuals


def calculate_accuracy(predictions, actuals, threshold=0.1):
    errors = np.abs(predictions - actuals)
    correct = np.sum(errors <= threshold)
    total = len(predictions)
    accuracy = (correct / total) * 100
    return accuracy, errors


def show_examples(x_data, y_data, predictions, actuals, num_examples=3):
    errors = np.abs(predictions - actuals)
    wrong_indices = np.argsort(errors)[-num_examples:][::-1]
    
    print(f"\nTop {num_examples} Worst Predictions:")
    print("-" * 70)
    for i, idx in enumerate(wrong_indices, 1):
        x_sample = x_data[idx]
        pred = predictions[idx]
        actual = actuals[idx]
        error = errors[idx]
        print(f"\nExample {i}:")
        print(f"  Input: Gender={x_sample[0]:.0f}, Age={x_sample[1]:.4f}, Height={x_sample[2]:.4f}")
        print(f"  Predicted: {pred:.6f}")
        print(f"  Actual: {actual:.6f}")
        print(f"  Error: {error:.6f} ({error/actual*100:.2f}% off)")


def main():
    num_epochs = 50

    print("Training...")
    for epoch in range(num_epochs):
        avg_loss = train_epoch()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}")

    print(f"\nFinal weights: {weights.value.flatten()}")
    print(f"Final bias: {bias.value.item():.6f}")

    print("\n" + "="*50)
    print("Evaluation on Test Set:")
    print("="*50)
    
    test_loss, test_mae, test_rmse, test_preds, test_actuals = evaluate(x_test, y_test)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    
    test_accuracy, test_errors = calculate_accuracy(test_preds, test_actuals, threshold=0.1)
    print(f"Test Accuracy (within 0.1): {test_accuracy:.2f}%")

    train_loss, train_mae, train_rmse, train_preds, train_actuals = evaluate(x_train, y_train)
    print(f"\nTrain Loss (MSE): {train_loss:.6f}")
    print(f"Train MAE: {train_mae:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}")
    
    train_accuracy, train_errors = calculate_accuracy(train_preds, train_actuals, threshold=0.1)
    print(f"Train Accuracy (within 0.1): {train_accuracy:.2f}%")

    print(f"\nTrain samples: {len(x_train)}, Test samples: {len(x_test)}")
    
    show_examples(x_test, y_test, test_preds, test_actuals, num_examples=3)


if __name__ == "__main__":
    main()
