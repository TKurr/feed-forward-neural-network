import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_training_history(history, save_path="training_history.png"):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure()

    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

def plot_weight_distribution(weights, layers):
    for i in layers:
        w = weights[i].flatten()
        plt.hist(w, bins=30)
        plt.title(f"Weight Distribution Layer {i}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.savefig(f"plot_weight_layer_{i}.png")
        plt.close()

def plot_gradient_distribution(grad_w, layers):
    for i in layers:
        g = grad_w[i].flatten()
        plt.hist(g, bins=30)
        plt.title(f"Gradient Distribution Layer {i}")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.savefig(f"plot_grad_layer_{i}.png")
        plt.close()