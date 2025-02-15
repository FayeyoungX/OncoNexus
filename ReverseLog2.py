import pandas as pd
import numpy as np
import os

# Use current directory
File_Path = os.path.dirname(os.path.abspath(__file__))

# Find all log2.tpm files
files = [f for f in os.listdir(File_Path) 
         if os.path.isfile(os.path.join(File_Path, f)) 
         and 'log2.tpm' in f]

for file in files:
    try:
        # Read and process file
        file_path = os.path.join(File_Path, file)
        df = pd.read_csv(file_path)
        
        # Separate first column (gene names) from numeric data
        gene_col = df.iloc[:, 0]  # Save first column
        numeric_data = df.iloc[:, 1:]  # Get numeric columns
        
        # Convert numeric columns to float
        numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
        
        # Reverse log2 transformation and handle negative values
        numeric_data = 2 ** numeric_data
        numeric_data = numeric_data.map(lambda x: max(x, 0.001) if pd.notnull(x) else 0.001)
        
        # Combine gene names with processed numeric data
        df = pd.concat([gene_col, numeric_data], axis=1)
        
        # Save processed file to same directory
        prefix = file.split('.')[0]
        new_name = os.path.join(File_Path, f'{prefix}.tpm.csv')
        df.to_csv(new_name, index=False)
        
        print(f'Successfully processed: {file}')
        
    except pd.errors.EmptyDataError:
        print(f'Error: {file} is empty or corrupted')
    except pd.errors.ParserError:
        print(f'Error: {file} has invalid format')
    except Exception as e:
        print(f'Unexpected error processing {file}: {str(e)}')
