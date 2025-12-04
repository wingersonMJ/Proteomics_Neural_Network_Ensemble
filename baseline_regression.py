import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold

from data_processing_pt2 import final_x, final_y

X = final_x
y = final_y

X_lim = X.iloc[:, 0:100]

X_np = np.asarray(X_lim)
y_np = np.asarray(y).ravel()

def mse(y_pred, y):
    loss = np.mean((y_pred - y)**2)
    return loss

def regression_cv(model, X, y):
    # set folds
    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    train_loss = []
    val_loss = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 
                                                 start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # fit model
        model.fit(X_train, y_train)

        # predict
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # get loss
        mse_train = mse(y_pred_train, y_train)
        mse_val = mse(y_pred_val, y_val)

        train_loss.append(mse_train)
        val_loss.append(mse_val)

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    print(f"Mean Train MSE: {train_loss.mean():.4f}")
    print(f"Mean Val MSE: {val_loss.mean():.4f}")

# model
linearModel = LinearRegression()
regression_cv(linearModel, X_np, y_np)

# try some regularization
alphas = [0.01, 0.5, 1.0, 2.0]
for a in alphas:
    ridgeModel = Ridge(alpha=a)
    print(f"\nAlpha: {a}")
    regression_cv(ridgeModel, X_np, y_np)

# re-fit best regularized model
ridgeModel = Ridge(alpha=2.0)
final_model = ridgeModel.fit(X_np, y_np)
y_pred = final_model.predict(X_np)

# plot y values
plt.figure()
sns.kdeplot(y_np, fill=True)
plt.xlabel("Target values")
plt.show()

# plot predictions vs actuals
plt.figure(figsize=(6,6))
plt.scatter(x=y_pred, y=y_np)
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Baseline Regression with Regularization")
plt.legend()
plt.tight_layout()
plt.savefig("./figs/baseline_regressor.jpg", dpi=300)
plt.show()