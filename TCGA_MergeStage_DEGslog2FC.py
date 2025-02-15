import pandas as pd
import os

#get all file names
files = [f for f in os.listdir('.') if os.path.isfile(f)]
#establish a dictionary to  save them
data_dict = {}

#ergodic all files, name and dataframe
for file in files:
    if '-' not in file or '.' not in file:
        continue
    try:
        prefix, rest = file.split('-')
        suffix = rest.split('.')[0]
        df = pd.read_csv(file, sep='\t', index_col=0)
    except Exception as e:
        print(f"Skipping file {file}: {str(e)}")
        continue
    if 'log2FoldChange' in df.columns:
        if prefix not in data_dict:
            data_dict[prefix] = df[['log2FoldChange']].rename(columns={'log2FoldChange':suffix})
        else:
            data_dict[prefix] = data_dict[prefix].merge(df[['log2FoldChange']].rename(columns={'log2FoldChange':suffix}), left_index=True, right_index=True, how='outer')
for prefix, df in data_dict.items():
    stages = ['StageI','StageII','StageIII','StageIV']
    for stage in stages:
        if stage not in df.columns:
            df[stage] = None
    df = df[stages]#reorganized
    df.to_csv(f'{prefix}_merged.csv', index=True)

print("Files merged!")
