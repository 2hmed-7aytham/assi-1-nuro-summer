import numpy as np

def tanh(x):
    return np.tanh(x)


np.random.seed(42)  
w1 = np.random.uniform(-0.5, 0.5, (2, 3))  
w2 = np.random.uniform(-0.5, 0.5, (3, 1))  


b1 = 0.5
b2 = 0.7

X = np.array([[1.0, 2.0]])  


print("=== Neural Network Forward Pass ===")
print(f"Input: {X}")
print(f"Weights W1:\n{w1}")
print(f"Weights W2:\n{w2}")
print(f"Bias b1: {b1}")
print(f"Bias b2: {b2}")


z1 = np.dot(X, w1) + b1
a1 = tanh(z1)
print(f"\nHidden layer input (z1): {z1}")
print(f"Hidden layer output (a1): {a1}")


z2 = np.dot(a1, w2) + b2
output = tanh(z2)

print(f"\nOutput layer input (z2): {z2}")
print(f"Final output: {output}")
print(f"Output shape: {output.shape}")