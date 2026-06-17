# Digit Identifier

A handwritten digit identifier built from scratch using Python and NumPy.

This project trains a simple neural network on MNIST-style digit data and predicts digits from a 28x28 image input. The model logic is implemented manually without using PyTorch or TensorFlow.

## Features

- Neural network built from scratch using NumPy
- Manual forward propagation
- ReLU activation
- Softmax output layer
- One-hot encoding
- Backpropagation
- Gradient descent
- Model parameter saving using `.npz`
- Local image-based digit prediction
- Popup result display

## Project Structure

```text
DigitIdentifier/
├── data/
│   └── mnist.npz
│
├── mainNeuralNetwork/
│   ├── main.py
│   └── app.py
│
├── digit.png
└── README.md
```

## How It Works

`main.py` trains the neural network and saves the learned parameters:

```text
w1, b1, w2, b2
```

These parameters are stored in:

```text
data/mnist.npz
```

`app.py` loads the saved model parameters, reads `digit.png`, converts it into a 28x28 normalized input, runs forward propagation, and displays the predicted digit in a popup.

## Installation

Install the required packages:

```bash
pip install numpy pandas pillow matplotlib
```

## Usage

Train the model:

```bash
python mainNeuralNetwork/main.py
```

Run digit prediction:

```bash
python mainNeuralNetwork/app.py
```

## Result

The model reached around 95% training accuracy.

## Notes

This project was built to understand the internal working of neural networks, including forward propagation, backpropagation, gradient descent, and parameter updates.