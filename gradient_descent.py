import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants and initial parameters
G = 10
learning_rate = 0.01
w = 10
b = 10


def v(x: float) -> float:
    """Target function."""
    return G * np.sqrt(2 * x / G)


def net(x: float, weight: float, bias: float) -> float:
    """Linear model."""
    return weight * x + bias


h = 10
print("target", v(h))

# Store the entire optimization trajectory
w_history = [w]
b_history = [b]
loss_history = [(net(h, w, b) - v(h)) ** 2]

for i in range(10):
    prediction = net(h, w, b)
    error = prediction - v(h)
    print(i, "iteration", "w", w, "b", b, "prediction", prediction, "error", error)

    # Gradient descent step
    w = w - learning_rate * h * error
    b = b - learning_rate * error

    # Save updated parameters and loss
    w_history.append(w)
    b_history.append(b)
    loss_history.append((net(h, w, b) - v(h)) ** 2)

# Grid for surface plot
w_vals = np.linspace(0, 10, 100)
b_vals = np.linspace(-10, 10, 100)
W, B = np.meshgrid(w_vals, b_vals)
Loss = (net(h, W, B) - v(h)) ** 2

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the optimization path
ax.plot(w_history, b_history, loss_history, "o-", color="red", linewidth=2)

# Plot the loss surface
ax.plot_surface(W, B, Loss, cmap="viridis", alpha=0.8)

ax.set_xlabel("Weight (w)")
ax.set_ylabel("Bias (b)")
ax.set_zlabel("Loss")
ax.set_title("Loss Function Surface")
plt.show()
