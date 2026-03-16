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


def plotTrainingHistoryComparison(history1, history2, label1="Model 1", label2="Model 2", savePath=None):
    # Plot perbandingan training dan validation loss untuk dua model (masing-masing satu plot)
    trainLoss1 = history1["train_loss"]
    valLoss1 = history1["val_loss"]
    trainLoss2 = history2["train_loss"]
    valLoss2 = history2["val_loss"]

    epochs = range(1, len(trainLoss1) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Model 1: Train + Val
    axes[0].plot(epochs, trainLoss1, label="Train", marker='o', markersize=3, color='blue')
    axes[0].plot(epochs, valLoss1, label="Val", marker='s', markersize=3, color='orange')
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"{label1} - Train vs Val Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Model 2: Train + Val
    axes[1].plot(epochs, trainLoss2, label="Train", marker='o', markersize=3, color='blue')
    axes[1].plot(epochs, valLoss2, label="Val", marker='s', markersize=3, color='orange')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title(f"{label2} - Train vs Val Loss", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

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

def plotWeightDistributionComparison(weights1, weights2, layers, label1="Model 1", label2="Model 2", savePath=None):
    # Plot perbandingan distribusi bobot untuk dua model
    numLayers = len(layers)
    fig, axes = plt.subplots(numLayers, 2, figsize=(12, 4 * numLayers))

    if numLayers == 1:
        axes = [axes]

    for idx, i in enumerate(layers):
        w1 = weights1[i].flatten()
        w2 = weights2[i].flatten()

        # Model 1
        axes[idx][0].hist(w1, bins=30, edgecolor='black', alpha=0.7, color='blue')
        axes[idx][0].set_title(f"{label1} - Weight Distribution Layer {i}", fontsize=12)
        axes[idx][0].set_xlabel("Weight Value", fontsize=11)
        axes[idx][0].set_ylabel("Frequency", fontsize=11)
        axes[idx][0].grid(True, alpha=0.3)

        # Model 2
        axes[idx][1].hist(w2, bins=30, edgecolor='black', alpha=0.7, color='red')
        axes[idx][1].set_title(f"{label2} - Weight Distribution Layer {i}", fontsize=12)
        axes[idx][1].set_xlabel("Weight Value", fontsize=11)
        axes[idx][1].set_ylabel("Frequency", fontsize=11)
        axes[idx][1].grid(True, alpha=0.3)

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=100, bbox_inches='tight')
    plt.show()


def plotGradientDistributionComparison(gradW1, gradW2, layers, label1="Model 1", label2="Model 2", savePath=None):
    # Plot perbandingan distribusi gradien untuk dua model
    numLayers = len(layers)
    fig, axes = plt.subplots(numLayers, 2, figsize=(12, 4 * numLayers))

    if numLayers == 1:
        axes = [axes]

    for idx, i in enumerate(layers):
        g1 = gradW1[i].flatten()
        g2 = gradW2[i].flatten()

        # Model 1
        axes[idx][0].hist(g1, bins=30, edgecolor='black', alpha=0.7, color='blue')
        axes[idx][0].set_title(f"{label1} - Gradient Distribution Layer {i}", fontsize=12)
        axes[idx][0].set_xlabel("Gradient Value", fontsize=11)
        axes[idx][0].set_ylabel("Frequency", fontsize=11)
        axes[idx][0].grid(True, alpha=0.3)

        # Model 2
        axes[idx][1].hist(g2, bins=30, edgecolor='black', alpha=0.7, color='red')
        axes[idx][1].set_title(f"{label2} - Gradient Distribution Layer {i}", fontsize=12)
        axes[idx][1].set_xlabel("Gradient Value", fontsize=11)
        axes[idx][1].set_ylabel("Frequency", fontsize=11)
        axes[idx][1].grid(True, alpha=0.3)

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=100, bbox_inches='tight')
    plt.show()

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