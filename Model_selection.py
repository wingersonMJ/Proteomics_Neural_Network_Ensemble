
import pandas as pd

import matplotlib.pyplot as plt

##########
# Load back in the results
##########
df = pd.read_csv("../Data/hyper_param_results.csv")
df.head()

# filter out high vals
df_filtered = df[df['Mean_fold_Vloss'] < 1000]
df_filtered.reset_index()

# scatter train and val loss
plt.figure(figsize=(8,8))
plt.scatter(x=df["Mean_fold_Tloss"], y=df["Mean_fold_Vloss"], s=20)
plt.title("Training vs Validation Loss for hyperparameter search")
plt.xlabel("Training Loss")
plt.ylabel("Validation Loss")
plt.savefig("./figs/search_results_all.jpg")
plt.show()

# scatter train and val loss
# filter out high vals
df_filtered = df[df['Mean_fold_Vloss'] < 1000]
df_filtered.reset_index()
# scatter train and val loss
plt.figure(figsize=(8,8))
plt.scatter(x=df_filtered["Mean_fold_Tloss"], y=df_filtered["Mean_fold_Vloss"], s=20)
plt.title("Training vs Validation Loss for hyperparameter search (filtered to Training Loss <1000)")
plt.xlabel("Training Loss")
plt.ylabel("Validation Loss")
plt.savefig("./figs/search_results_filtered.jpg")
plt.show()

# scatter train and val loss
# filter out high vals
df_possible = df[df['Mean_fold_Vloss'] < 300]
df_possible = df_possible[df_possible['Mean_fold_Tloss'] < 300]
df_possible.reset_index()
# scatter train and val loss
plt.figure(figsize=(8,8))
plt.scatter(x=df_possible["Mean_fold_Tloss"], y=df_possible["Mean_fold_Vloss"], s=50)
plt.title("Training vs Validation Loss for hyperparameter search (filtered to Training and Val Loss <300)")
plt.xlabel("Training Loss")
plt.ylabel("Validation Loss")
plt.savefig("./figs/search_results_possible.jpg")
plt.show()

# identify model combo index values
df_id = df[df['Mean_fold_Vloss'] < 280]
df_id = df_id[df_id['Mean_fold_Tloss'] < 280]
df_id.reset_index()
# scatter train and val loss
plt.figure(figsize=(8,8))
plt.scatter(x=df_id["Mean_fold_Tloss"], y=df_id["Mean_fold_Vloss"], 
            s=(df_id["SD_fold_Tloss"]**1.75),
            c=df_id["SD_fold_Vloss"])
for i, (x, y) in df_id[["Mean_fold_Tloss", "Mean_fold_Vloss"]].iterrows():
    plt.text(x, y, str(i), fontsize=8, ha="center", va="center", fontweight="bold", color="white")
plt.title("Model index values - selecting best model params")
plt.xlabel("Training Loss")
plt.ylabel("Validation Loss")
plt.colorbar()
plt.savefig("./figs/search_results_indexed_models.jpg", dpi=300)
plt.show()

# filter to best param combos
best_models_idx = [436, 620, 112, 296, 180,
                   472, 504, 149, 181, 184]
best_params = df[df['Unnamed: 0'].isin(best_models_idx)]
print(best_params)
# plot best param combos
plt.figure(figsize=(9,6))
plt.scatter(x=best_params["Mean_fold_Tloss"], 
            y=best_params["Mean_fold_Vloss"],
            s=400)
for i, (x, y) in best_params[["Mean_fold_Tloss", "Mean_fold_Vloss"]].iterrows():
    plt.text(x, y, str(i), fontsize=8, ha="center", va="center", fontweight="bold", color="white")
plt.plot([210, 270], [210, 270],
         linestyle="--", color="dimgray")
plt.title("Models with the best hyperparameters based on CV performance")
plt.xlabel("Training Loss")
plt.ylabel("Validation Loss")
plt.savefig("./figs/search_results_best_models.jpg", dpi=300)
plt.show()

#################
# show model configs
print(f"Number of Models: {len(best_params)}\n")
print(
    "Index | Mean_fold_Tloss | SD_fold_Tloss | Mean_fold_Vloss | "
    "----------------- batch_size | momentum | optimizer | "
    "max_norm | epochs_ran | dropout_p"
)
for idx, row in best_params.iterrows():
    print(
        f'{row["Unnamed: 0"]} | '
        f'{row["Mean_fold_Tloss"]} | '
        f'{row["SD_fold_Tloss"]} | '
        f'{row["Mean_fold_Vloss"]} | '
        f'{row["SD_fold_Vloss"]} | '
        f'{row["batch_size"]} | '
        f'{row["momentum"]} | '
        f'{row["optimizer"]} | '
        f'{row["max_norm"]} | '
        f'{row["epochs_ran"]:.0f} | '
        f'{row["dropout_p"]}'
    )

# add LR to this...
lr = [0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.001, 0.001, 0.001, 0.0001]

best_params["lr"] = lr 