import matplotlib.pyplot as plt


def plotTrainingHistory(history, savePath=None):
    # Plot training dan validation loss per epoch
    trainLoss = history["train_loss"]
    valLoss = history["val_loss"]

    epochs = range(1, len(trainLoss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, trainLoss, label="Train Loss", marker='o', markersize=3)
    ax.plot(epochs, valLoss, label="Validation Loss", marker='s', markersize=3)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training History", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=100, bbox_inches='tight')
    plt.show()


def plotWeightDistribution(weights, layers):
    # Plot histogram distribusi bobot untuk layer tertentu
    for i in layers:
        w = weights[i].flatten()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(w, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f"Weight Distribution Layer {i}", fontsize=12)
        ax.set_xlabel("Weight Value", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot_weight_layer_{i}.png", dpi=100, bbox_inches='tight')
        plt.close()


def plotGradientDistribution(gradW, layers):
    # Plot histogram distribusi gradien untuk layer tertentu
    for i in layers:
        g = gradW[i].flatten()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(g, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.set_title(f"Gradient Distribution Layer {i}", fontsize=12)
        ax.set_xlabel("Gradient Value", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot_grad_layer_{i}.png", dpi=100, bbox_inches='tight')
        plt.close()