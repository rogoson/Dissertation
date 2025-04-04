import numpy as np
import matplotlib.pyplot as plt


# Define the normalization function
def normaliseValue(value):
    return np.sign(value) * np.log1p(np.abs(value))


# Create a range of values from -1 to 1
x = np.linspace(-100, 100, 500)

# Apply the normalization function
y = normaliseValue(x)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="normaliseValue(value)", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--", label="y = 0")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--", label="x = 0")
plt.title("Behavior of normaliseValue Function")
plt.xlabel("Input Value")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.show()
