import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-poster')

df = pd.read_csv("..\\Data\\final_df.csv")

# drop extra columns
df.drop(columns=['index', 'SampleName', 'Unnamed: 0'], inplace=True)

start_idx = df.columns.get_loc('10043-31')
df = df.iloc[:, :start_idx]

# fix missingness 
missingness = df.isna().sum(axis=1)

high_missingness = [30, 43, 50, 54]
df = df.drop(index=high_missingness)

df.isna().sum(axis=1)

# separate out x and y
x = df.drop(columns='Max_Curve')
y = df['Max_Curve']

# encode sex
x['Sex'] = x['Sex'].map({'Male': 0, 'Female': 1})

# k-means imputation
feature_missingness = x.isna().sum()
feature_missingness = [f for f in feature_missingness if f > 0]

imputer = KNNImputer(n_neighbors=5)
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns, index=x.index)

x.isna().sum()
reduced_x = x.copy()
reduced_x = reduced_x.iloc[:, 0:25]

final_x = x.copy()
final_y = y.copy()

# plots outlined below. Commented to they don't re-run 
# every time I import final_x, final_y
"""
# kde plot
plt.figure(figsize=(8,6))
sns.histplot(final_y, kde=True, alpha=0.5, bins=8)
plt.title("Distribution of Target Variable")
plt.tight_layout()
plt.savefig("./figs/target_distribution.jpg", dpi=300)
plt.close()

# correlation map
cor_matrix = reduced_x.corr()
cor_matrix = np.abs(cor_matrix)

plt.figure(figsize=(6,6))
ax = sns.heatmap(cor_matrix,
            annot=False,
            cmap="Blues",
            linecolor="white",
            linewidths=0.7
)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.title("Correlation Matrix for Predictors")
plt.tight_layout()
plt.savefig("./figs/corr_map.jpg", dpi=300)
plt.close()
"""
