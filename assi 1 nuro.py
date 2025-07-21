import numpy as np

# Define tanh activation function
def tanh(x):
    return np.tanh(x)

# Initialize weights randomly from interval [-0.5, 0.5]
np.random.seed(42)  # For reproducible results
w1 = np.random.uniform(-0.5, 0.5, (2, 3))  # 2 inputs, 3 hidden neurons
w2 = np.random.uniform(-0.5, 0.5, (3, 1))  # 3 hidden, 1 output

# Set biases
b1 = 0.5
b2 = 0.7

# Sample input data
X = np.array([[1.0, 2.0]])  # 1 sample, 2 features

# Forward pass
print("=== Neural Network Forward Pass ===")
print(f"Input: {X}")
print(f"Weights W1:\n{w1}")
print(f"Weights W2:\n{w2}")
print(f"Bias b1: {b1}")
print(f"Bias b2: {b2}")

# Hidden layer
z1 = np.dot(X, w1) + b1
a1 = tanh(z1)
print(f"\nHidden layer input (z1): {z1}")
print(f"Hidden layer output (a1): {a1}")

# Output layer
z2 = np.dot(a1, w2) + b2
output = tanh(z2)

print(f"\nOutput layer input (z2): {z2}")
print(f"Final output: {output}")
print(f"Output shape: {output.shape}")