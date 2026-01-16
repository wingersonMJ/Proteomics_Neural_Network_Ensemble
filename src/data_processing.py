import pandas as pd
import numpy as np
import pyreadr

# load in proteomics data from sample and validation set
proteomics = pd.read_csv("..\\Data\\Proteomics_Resid.txt", sep='\t')
val_proteomics = pyreadr.read_r(
    "..\\Data\\Proteomics_Resid_Validation.RDS")
val_proteomics = val_proteomics[None]

# switch columns to rows and vice versa 
proteomics = proteomics.T
proteomics.reset_index(inplace=True)

val_proteomics = val_proteomics.T
val_proteomics.reset_index(inplace=True)

# adjust column names
proteomics.columns = (
    proteomics.columns
    .str.replace('seq.', '', regex=False)
    .str.replace('.', '-', regex=False)
    .str.strip()
)

# load in demographics 
demograph = pd.read_csv("..\\Data\\Deidentified_Phenotype.csv")
val_demograph = pd.read_csv("..\\Data\\Validation_Phenotype.csv")

# adjust val_demographics sample names
val_proteomics['index'] = (
    val_proteomics['index']
    .str.replace('.', '-', regex=False)
)

# merge with demographics
merged_main = pd.merge(
    proteomics, demograph, left_on='index', 
    right_on='fam_ind_id', how='outer')

merged_val = pd.merge(
    val_proteomics, val_demograph, left_on='index',
    right_on='SampleName', how='outer')

# align column names
merged_main = merged_main.rename(
    columns={'fam_ind_id': 'SampleName', 
            'Age_Surgery': 'Age'})

# drop obs, 
merged_main = merged_main.drop(
    columns=[
        'Obs', 'include', 'reason', 'NHW_Final_YN', 
        'Curve_Pattern_Final'])
merged_val = merged_val.drop(columns=['Race', 'Ethnicity'])

# merge
df = pd.concat(
    [merged_main, merged_val],
    axis=0, join='outer', ignore_index=False)

# save
df.to_csv("..\\Data\\final_df.csv")
