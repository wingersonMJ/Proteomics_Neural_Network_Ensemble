import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from data_processing_pt2 import final_x, final_y

plt.style.use("seaborn-v0_8-poster")

X = final_x
y = final_y

X_np = np.asarray(X)
y_np = np.asarray(y).ravel()

def mse(y_pred, y):
    loss = np.mean((y_pred - y)**2)
    return loss

def regression_cv(model, X, y):

    # set folds
    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    # track losses, predictions, and actuals for each fold
    train_loss = []
    val_loss = []
    val_preds = []
    val_actuals = []

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

        # track loss
        train_loss.append(mse_train)
        val_loss.append(mse_val)

        # track preds and actuals
        val_preds.append(y_pred_val)
        val_actuals.append(y_val)

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    print(f"Mean Train MSE: {train_loss.mean():.4f}")
    print(f"Mean Val MSE: {val_loss.mean():.4f}")
    mean_tLoss = train_loss.mean()
    mean_vLoss = val_loss.mean()
    std_tLoss = train_loss.std()
    std_vLoss = val_loss.std()
    return (mean_tLoss, mean_vLoss, 
            std_tLoss, std_vLoss,
            val_preds, val_actuals)

# model
linearModel = LinearRegression()
regression_cv(linearModel, X_np, y_np)

# try some regularization
alphas = [100, 200, 300, 400, 500, 600, 700, 
          800, 900, 1000, 2000, 3000, 4000, 
          5000, 6000, 7000, 8000, 9000, 
          10000, 15000, 20000, 30000]
tLoss_alphas = []
vLoss_alphas = []
for a in alphas:
    ridgeModel = Ridge(alpha=a)
    print(f"\nAlpha: {a}")
    tLoss, vLoss, std_vLoss, std_tLoss, p, a = regression_cv(
        ridgeModel, X_np, y_np)
    tLoss_alphas.append(tLoss)
    vLoss_alphas.append(vLoss)

plt.figure(figsize=(8,7))
plt.plot(alphas, tLoss_alphas, 
         label="training loss")
plt.plot(alphas, vLoss_alphas,
         label="validation loss")
plt.legend()
plt.xlabel("Alpha in Ridge Normalization")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss - Baseline Model")
plt.savefig("./figs/ridge_alphas.jpg", dpi=300)
plt.show()

# re-fit best regularized model
ridgeModel = Ridge(alpha=500)
final_model = ridgeModel.fit(X_np, y_np)
t, v, t_std, v_std, p, a = regression_cv(ridgeModel, X_np, y_np)

# concat each fold preds and actuals
all_val_preds = np.concatenate(p)
all_val_actuals = np.concatenate(a)

# plot predictions vs actuals
sns.jointplot(x=all_val_preds, y=all_val_actuals)
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Baseline Regression")
plt.tight_layout()
plt.savefig("./figs/baseline_regressor.jpg", dpi=300)
plt.show()

sns.jointplot(x=all_val_preds, y=all_val_actuals, 
              kind='kde', fill=True)
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Baseline Regression")
plt.tight_layout()
plt.savefig("./figs/baseline_regressor_kde.jpg", dpi=300)
plt.show()

# final baseline performance
print(f"Train loss: {t:.1f}\n"
    f"Val loss: {v:.1f}\n"
    f"Train SD: {t_std:.2f}\n"
    f"Val SD: {v_std:.2f}"
)
