import pandas as pd 
import numpy as np

from sklearn.impute import KNNImputer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(".\\Data\\final_df.csv")

# drop extra columns
df.drop(columns=['index', 'SampleName', 'Unnamed: 0'], inplace=True)

start_idx = df.columns.get_loc('10043-31')
print(start_idx)
df = df.iloc[:, :start_idx]

# fix missingness 
missingness = df.isna().sum(axis=1)
print(missingness)

high_missingness = [30, 43, 50, 54]
df = df.drop(index=high_missingness)

df.isna().sum(axis=1)

# separate out x and y
x = df.drop(columns='Max_Curve')
y = df['Max_Curve']

# encode sex
x['Sex'] = x['Sex'].map({'Male': 0, 'Female': 1})

# k-means imputation
imputer = KNNImputer(n_neighbors=500)
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns, index=x.index)

x.isna().sum()

x.shape
y.shape

final_x = x.copy()
final_y = y.copy()