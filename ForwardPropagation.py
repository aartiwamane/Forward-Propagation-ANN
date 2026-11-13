
import numpy as np


# ------------------------------------------------------------
# Display Section Header
# ------------------------------------------------------------

def Display_Section(step, title):

    print("\n" + "=" * 65)
    print(f"Step {step}: {title}")
    print("=" * 65)


# ------------------------------------------------------------
# Step 1: Input Dataset
# ------------------------------------------------------------

Display_Section(1, "Neural Network Input")

X = np.array([
    [0.5, 0.8],
    [0.2, 0.4],
    [0.9, 0.7]
])

print("Input Data:")
print(X)


# ------------------------------------------------------------
# Step 2: Initialize Weights and Biases
# ------------------------------------------------------------

Display_Section(2, "Weight and Bias Initialization")

W1 = np.array([
    [0.2, 0.4],
    [0.3, 0.5]
])

b1 = np.array([0.1, 0.2])

W2 = np.array([
    [0.6],
    [0.7]
])

b2 = np.array([0.1])

print("Hidden Layer Weights:")
print(W1)

print("\nHidden Layer Bias:")
print(b1)

print("\nOutput Layer Weights:")
print(W2)

print("\nOutput Layer Bias:")
print(b2)


# ------------------------------------------------------------
# Step 3: Define Activation Function
# ------------------------------------------------------------

Display_Section(3, "Sigmoid Activation Function")


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


print("Sigmoid(0):", sigmoid(0))


# ------------------------------------------------------------
# Step 4: Hidden Layer Forward Propagation
# ------------------------------------------------------------

Display_Section(4, "Hidden Layer Computation")

Z1 = np.dot(X, W1) + b1

A1 = sigmoid(Z1)

print("Hidden Layer Weighted Sum:")
print(Z1)

print("\nHidden Layer Activation:")
print(A1)


# ------------------------------------------------------------
# Step 5: Output Layer Forward Propagation
# ------------------------------------------------------------

Display_Section(5, "Output Layer Computation")

Z2 = np.dot(A1, W2) + b2

A2 = sigmoid(Z2)

print("Output Layer Weighted Sum:")
print(Z2)

print("\nFinal Network Predictions:")
print(A2)


# ------------------------------------------------------------
# Step 6: Display Predictions
# ------------------------------------------------------------

Display_Section(6, "Final Prediction Results")

for index, prediction in enumerate(A2):

    print(f"Input Sample {index + 1}: {X[index]}")

    print(f"Predicted Output: {prediction[0]:.4f}")

    print("-" * 40)


# ------------------------------------------------------------
# Step 7: Understand Network Architecture
# ------------------------------------------------------------

Display_Section(7, "Neural Network Architecture")

print("Input Layer  : 2 Neurons")
print("Hidden Layer : 2 Neurons")
print("Output Layer : 1 Neuron")

print("\nForward Propagation Completed Successfully!")
