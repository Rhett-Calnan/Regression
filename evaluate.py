import numpy as np
import matplotlib.pyplot as plt

def mae(y, prediction):
    """Mean absolute error, in dollars."""
    return float(np.mean(np.abs(prediction - y)))

def rmse(y, prediction):
    """Root mean squared error, in dollars."""
    return float(np.sqrt(np.mean((prediction - y) ** 2)))

def plot_results(area_test, y_test, prediction, history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Parameter updates", ylabel="Cost ($ squared)", title="Training cost")
    order = np.argsort(area_test)
    axes[1].scatter(area_test, y_test, alpha=0.65, label="Test prices")
    axes[1].plot(area_test[order], prediction[order], color="orange", label="Learned line")
    axes[1].set(xlabel="Floor area (m2)", ylabel="Price ($)", title="Unseen examples")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)