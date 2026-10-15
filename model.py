import numpy as np

def predict(X, w, b):
    """Return one predicted price per scaled floor area."""
    return w * X + b

def compute_cost(X, y, w, b):
    """Return half the mean squared error."""
    error = predict(X, w, b) - y
    return float(np.mean(error ** 2) / 2)

def compute_gradients(X, y, w, b):
    """Return the cost derivatives with respect to w and b."""
    error = predict(X, w, b) - y
    dw = np.mean(error * X)
    db = np.mean(error)
    return float(dw), float(db)

def fit(X, y, learning_rate=0.1, iterations=3000):
    """Learn parameters using full-batch gradient descent."""
    w, b = 0.0, 0.0
    history = [compute_cost(X, y, w, b)]
    for _ in range(iterations):
        dw, db = compute_gradients(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        cost = compute_cost(X, y, w, b)
        if not np.isfinite(cost):
            raise ValueError("Loss diverged; lower learning rate")
        history.append(cost)
    return w, b, np.array(history)