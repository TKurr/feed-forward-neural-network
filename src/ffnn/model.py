import numpy as np
import pickle

class FFNN:
    def __init__(self, layer_sizes, activations, loss, initializer, regularizer=None):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_fn = loss
        self.initializer = initializer
        self.regularizer = regularizer

        self.weights = []
        self.biases = []
        self.grad_w = []
        self.grad_b = []

        self._initialize_weights()

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
            a = self.activations[i].forward(z)
            self.z_values.append(z)
            self.a_values.append(a)
        return self.a_values[-1]

    def backward(self, y_true):
        # hitung gradien dari loss ke weight dan bias setiap layer
        self.grad_w = []
        self.grad_b = []

        m = y_true.shape[0]

        delta = self.loss_fn.backward(y_true, self.a_values[-1])

        for i in reversed(range(len(self.weights))):
            delta *= self.activations[i].backward(self.z_values[i])

            dW = (self.a_values[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            if self.regularizer:
                dW += self.regularizer.gradient(self.weights[i])

            self.grad_w.insert(0, dW)
            self.grad_b.insert(0, db)

            delta = delta @ self.weights[i].T
            
    def update(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.grad_w[i]
            self.biases[i] -= lr * self.grad_b[i]

    def fit(self, X_train, y_train,
            X_val=None, y_val=None,
            epochs=100,
            lr=0.01,
            batch_size=None,
            verbose=1):

        history = {
            "train_loss": [],
            "val_loss": []
        }

        n_samples = X_train.shape[0]

        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0

            for start in range(0, n_samples, batch_size):

                end = start + batch_size

                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # forward pass
                y_pred = self.forward(X_batch)

                # hitung loss
                loss = self.loss_fn.forward(y_batch, y_pred)

                # backward propagation
                self.backward(y_batch)

                # update
                self.update(lr)

                epoch_loss += loss * len(X_batch)

            epoch_loss /= n_samples

            history["train_loss"].append(epoch_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.loss_fn.forward(y_val, val_pred)
                history["val_loss"].append(val_loss)
            else:
                val_loss = None

            if verbose == 1:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f}")

        return history

    def predict(self, X):
        return self.forward(X)

    def save(self, path):
        # save model
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)