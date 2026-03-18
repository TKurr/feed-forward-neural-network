# Feed Forward Neural Network

Feed-forward neural network from scratch in NumPy.

> Feedforward Neural Network (FNN) is a type of artificial neural network in which information flows in a single direction i.e from the input layer through hidden layers to the output layer without loops or feedback. It is mainly used for pattern recognition tasks like image and speech classification.

![Dataset](<https://img.shields.io/badge/Dataset-Student%20Placement%20(Binary)-blue>)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-orange)

## **Overview**

Implementing Feed Forward Neural Network from scratch (without Tensorflow/Pytorch), trained model for student placement prediction. Other important parts of implementation in this repository includes:

- Activation functions implementation: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU
- EDA, Cleaning, and Preprocessing
- Optimizers: GD, Adam
- Regularization: L1, L2
- Initialization: Xavier, He, Uniform, Normal
- Loss: MSE, BinaryCrossEntropy, CategoricalCrossEntropy
- Layer normalization: RMSNorm

## **Setup**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Project structure:**

```
feed-forward-neural-network
├── data
│   ├── datasetml_2026.csv
│   └── processed/cleandataset.csv
├── doc
├── LICENSE
├── notebooks
│   ├── 0-eda-preprocess.ipynb
│   ├── 1-pengujian-terpisah-ffnn.ipynb
│   ├── 2-perbandingan-optimizers.ipynb
│   ├── 3-eksperimen-normalisasi.ipynb
│   └── models/processedData.pkl
├── README.md
├── requirements.txt
├── src
│   ├── ffnn
│   ├── main.py
│   ├── plot_grad_layer_0.png
│   ├── plot_grad_layer_1.png
│   ├── plot_grad_layer_2.png
│   ├── plot_weight_layer_0.png
│   ├── plot_weight_layer_1.png
│   ├── plot_weight_layer_2.png
│   └── training_history.png
└── venv
```

### Bonus 🙏

- Adam (Adaptive Moment Estimation) Optimizer implementation
- RMSNorm normalization
- Xavier and He Initialization
- LeakyELU and ELU activations

---

### Author

| NIM      | Name                |
| :------- | :------------------ |
| 13523148 | Andrew Tedjapratama |
| 13523154 | Theo Kurniady       |
| 13523158 | Lukas Raja Agripa   |

## Acknowledgements

- Machine Learning Course Lecturer, Bandung Institute of Technology, 2026
- Machine Learning Teaching Assistants, Bandung Institute of Technology, 2026
