from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from data import make_data, split_data
from model import fit, predict
from evaluate import mae, rmse, plot_results

def main():
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    area, price = make_data()
    area_train, area_test, y_train, y_test = split_data(area, price)
    # X is in hundreds of m2; y remains in dollars.
    X_train = area_train / 100.0
    X_test = area_test / 100.0
    w, b, history = fit(X_train, y_train)
    pred_train = predict(X_train, w, b)
    pred_test = predict(X_test, w, b)
    # Baseline uses ONLY training targets.
    baseline = np.full(y_test.shape, y_train.mean())
    reference = LinearRegression()
    reference.fit(X_train.reshape(-1, 1), y_train)
    pred_reference = reference.predict(X_test.reshape(-1, 1))
    print(f"Train/test examples: {len(y_train)}/{len(y_test)}")
    print(f"w: {w:,.2f}; b: {b:,.2f}")
    print(f"Slope per m2: ${w / 100:,.2f}")
    print(f"Training MAE: ${mae(y_train, pred_train):,.2f}")
    print(f"Baseline test MAE: ${mae(y_test, baseline):,.2f}")
    print(f"Your test MAE: ${mae(y_test, pred_test):,.2f}")
    print(f"Your test RMSE: ${rmse(y_test, pred_test):,.2f}")
    print(f"Sklearn test MAE: ${mae(y_test, pred_reference):,.2f}")
    gap = np.max(np.abs(pred_test - pred_reference))
    print(f"Maximum prediction gap vs sklearn: ${gap:.4f}")
    new_area = np.array([120.0])
    new_price = predict(new_area / 100.0, w, b)[0]
    print(f"Prediction for 120 m2: ${new_price:,.2f}")
    plot_results(area_test, y_test, pred_test, history,
    output / "results.png")

if __name__ == "__main__":
    main()