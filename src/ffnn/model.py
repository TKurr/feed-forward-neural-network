import numpy as np
import pickle
import matplotlib.pyplot as plt
from .utils.optimizer import GradientDescent

class FFNN:
    def __init__(self, layer_sizes, activations, loss, initializer, regularizer=None, normalizers=None):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_fn = loss
        self.initializer = initializer
        self.regularizer = regularizer
        self.normalizers = normalizers

        self.weights = []
        self.biases = []
        self.grad_w = []
        self.grad_b = []

        self._validateArchitecture() # validasi layer size sama length activations match transformasi layer
        self._initialize_weights()

    def _validateArchitecture(self) -> None: 
        numLayers = len(self.layer_sizes) - 1
        if len(self.activations) != numLayers:
            raise ValueError(
                f"Jumlah activations ({len(self.activations)}) "
                f"harus sama dengan jumlah layer transformasi ({numLayers})"
            )

    def _initialize_weights(self):
        # Inisialisasi weight dan bias setiap layer
        for i in range(len(self.layer_sizes)-1):
            W = self.initializer.initialize((self.layer_sizes[i], self.layer_sizes[i+1]))
            b = self.initializer.initialize((1, self.layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        # forward pass dari input sampe jadi output
        self.z_values = []
        self.a_values = [X]
        for i in range(len(self.weights)):
            z = self.a_values[-1] @ self.weights[i] + self.biases[i]
            if self.normalizers and i < len(self.normalizers):
                z = self.normalizers[i].forward(z)
            a = self.activations[i].forward(z)
            self.z_values.append(z)
            self.a_values.append(a)
        return self.a_values[-1]

    def backward(self, y_true) -> None:
        # Hitung gradien dari loss ke weight dan bias setiap layer dengan chain rule
        # Untuk Softmax+CCE, activation backward tidak dikali karena CCE backward udah include
        self.grad_w = []
        self.grad_b = []

        m = y_true.shape[0]

        delta = self.loss_fn.backward(y_true, self.a_values[-1])

        for i in reversed(range(len(self.weights))):
            activationName = self.activations[i].__class__.__name__
            if not (activationName == 'Softmax'):
                delta *= self.activations[i].backward(self.z_values[i])
            if self.normalizers and i < len(self.normalizers):
                delta = self.normalizers[i].backward(delta)
                
            dW = (self.a_values[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            if self.regularizer:
                dW += self.regularizer.gradient(self.weights[i])

            self.grad_w.insert(0, dW)
            self.grad_b.insert(0, db)

            delta = delta @ self.weights[i].T
            

    def fit(self, X_train, y_train,
            X_val=None, y_val=None,
            epochs=100,
            lr=0.01,
            batch_size=None,
            optimizer=None,
            verbose=1):
        # Training loop mini-batch GD. nnti return history train_loss dan val_loss per epoch

        history = {
            "train_loss": [],
            "val_loss": []
        }

        n_samples = X_train.shape[0]

        if batch_size is None:
            batch_size = n_samples

        # Set default optimizer kalau None
        if optimizer is None:
            optimizer = GradientDescent(lr=lr)

        for epoch in range(epochs):

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):

                end = start + batch_size

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                y_pred = self.forward(X_batch)
                loss = self.loss_fn.forward(y_batch, y_pred)

                if self.regularizer:
                    regLoss = 0.0
                    for w in self.weights:
                        if 'L1' in self.regularizer.__class__.__name__:
                            regLoss += np.sum(np.abs(w))
                        else:
                            regLoss += np.sum(w**2)
                    loss += self.regularizer.lam * regLoss / n_samples

                self.backward(y_batch)
                optimizer.update_parameters(self.weights, self.biases, self.grad_w, self.grad_b)

                if self.normalizers:
                    for norm in self.normalizers:
                        if hasattr(norm, 'update_params'):
                            norm.update_params(lr)

                epoch_loss += loss * len(X_batch)

            epoch_loss /= n_samples

            history["train_loss"].append(epoch_loss)

            valLoss = None
            if X_val is not None:
                valPred = self.forward(X_val)
                valLoss = self.loss_fn.forward(y_val, valPred)

                if self.regularizer:
                    regVal = 0.0
                    for w in self.weights:
                        if 'L1' in self.regularizer.__class__.__name__:
                            regVal += np.sum(np.abs(w))
                        else:
                            regVal += np.sum(w**2)
                    valLoss += self.regularizer.lam * regVal / X_val.shape[0]

                history["val_loss"].append(valLoss)

            if verbose == 1:
                if valLoss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.6f} - val_loss: {valLoss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.6f}")

        return history

    def plotWeightDistribution(self, layerIndices=None, savePath=None) -> None:
        # Plot histogram distribusi bobot dan bias untuk layer pilihan

        if layerIndices is None:
            layerIndices = list(range(len(self.weights)))

        numLayers = len(layerIndices)
        numCols = min(3, numLayers)
        numRows = (numLayers + numCols - 1) // numCols

        fig, axes = plt.subplots(numRows, numCols, figsize=(5*numCols, 4*numRows))
        if numRows == 1 and numCols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, layerIdx in enumerate(layerIndices):
            weights = self.weights[layerIdx].flatten()
            biases = self.biases[layerIdx].flatten()
            allParams = np.concatenate([weights, biases])

            axes[idx].hist(allParams, bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Weight+Bias Layer {layerIdx}')
            axes[idx].set_xlabel('Parameter Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

        for idx in range(numLayers, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if savePath:
            plt.savefig(savePath, dpi=100, bbox_inches='tight')
        plt.show()

    def plotGradientDistribution(self, layerIndices=None, savePath=None):
        # fn plotGradientDistribution(layerIndices, savePath) -> None
        # Plot histogram distribusi gradien bobot dan bias untuk layer pilihan

        if layerIndices is None:
            layerIndices = list(range(len(self.grad_w)))

        numLayers = len(layerIndices)
        numCols = min(3, numLayers)
        numRows = (numLayers + numCols - 1) // numCols

        fig, axes = plt.subplots(numRows, numCols, figsize=(5*numCols, 4*numRows))
        if numRows == 1 and numCols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, layerIdx in enumerate(layerIndices):
            gradW = self.grad_w[layerIdx].flatten()
            gradB = self.grad_b[layerIdx].flatten()
            allGrads = np.concatenate([gradW, gradB])

            axes[idx].hist(allGrads, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[idx].set_title(f'Gradient Layer {layerIdx}')
            axes[idx].set_xlabel('Gradient Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

        for idx in range(numLayers, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if savePath:
            plt.savefig(savePath, dpi=100, bbox_inches='tight')
        plt.show()

    def predict(self, X):
        # Forward pass tanpa training, return prediksi output
        return self.forward(X)

    def save(self, path):
        # Simpan model ke file pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        # Load model dari file pickle
        with open(path, "rb") as f:
            return pickle.load(f)