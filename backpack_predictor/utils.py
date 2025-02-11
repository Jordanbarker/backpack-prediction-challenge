
import pandas as pd
from itertools import combinations

from .features import target

from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder


def prepare_data(df: pd.DataFrame, is_train: bool = True):
    """
    Prepares the dataset for training or testing by renaming columns, handling missing values,
    converting categorical and numerical features, and creating new features.
    
    Args:
        df (pd.DataFrame): The input dataframe (train or test).
        is_train (bool): Indicates if the dataframe is training data (default is True).
        
    Returns:
        pd.DataFrame: The processed dataframe.
    """
    
    # Define the column names
    columns = [
        'id', 'brand', 'material', 'size', 'compartments', 
        'laptop_compartment', 'is_waterproof', 'style', 'color', 
        'weight_capacity'
    ]
    
    if is_train:
        columns.append('price')
    
    df.columns = columns
    
    if is_train:
        df = df.drop(columns='id')
    
    # Define the mapping for Size conversion
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    # df["size_int"] = df["size"].map(size_mapping).fillna(0).astype(int)
    df["size"] = df["size"].map(size_mapping).fillna(-1)
    
    # Handle weight capacity
    df['weight_capacity'] = df['weight_capacity'].fillna(0)
    # df['weight_capacity_int'] = df['weight_capacity'].astype(int)
    # df['weight_capacity_size'] = df['weight_capacity'] * df['size_int']
    
    # Convert categorical columns
    df['compartments'] = df['compartments'].astype('category')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df[cat_cols] = df[cat_cols].astype('category')
    
    # Convert boolean columns to integer type
    df['laptop_compartment'] = df['laptop_compartment'].cat.codes.fillna(-1).astype(int)
    df['is_waterproof'] = df['is_waterproof'].cat.codes.fillna(-1).astype(int)
    
    return df


# def preprocess_weight_capacity(train_df, test_df, n_bins=5, target=target):
#     """
#     Function to bin 'weight_capacity' and apply Target Encoding based on the target column.
    
#     Parameters:
#     train_df (pd.DataFrame): Training dataframe containing 'weight_capacity'.
#     test_df (pd.DataFrame): Test dataframe containing 'weight_capacity'.
#     target_column (str): Target variable for encoding.
#     n_bins (int): Number of bins for discretization.
    
#     Returns:
#     pd.DataFrame, pd.DataFrame: Transformed train and test DataFrames.
#     """
#     # Apply KBinsDiscretizer to bin 'weight_capacity'
#     bins_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
#     train_df['binned_weight_capacity'] = bins_discretizer.fit_transform(train_df[['weight_capacity']])
#     test_df['binned_weight_capacity'] = bins_discretizer.transform(test_df[['weight_capacity']])

#     # Apply TargetEncoder to encode the binned values based on the target_column
#     target_encoder = TargetEncoder(target_type="continuous")
#     train_df['encoded_weight_capacity'] = target_encoder.fit_transform(train_df[['binned_weight_capacity']], train_df[target])
#     test_df['encoded_weight_capacity'] = target_encoder.transform(test_df[['binned_weight_capacity']])
    
#     return train_df, test_df


def target_encoding(
    train_df: pd.DataFrame,
    cat_cols: list,
    target: str = target,
    test_df: pd.DataFrame = None,
    interactions: bool = True
):
    # Make copies to avoid mutating original data
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None
    
    encoded_cols = []

    # --- Encode each individual categorical column with TargetEncoder ---
    for col in cat_cols:
        # Initialize a fresh TargetEncoder for each column
        te = TargetEncoder(target_type="continuous")
        
        # Fit on the training data
        train_encoded = te.fit_transform(train_df[[col]], train_df[target])
        train_encoded_col = f"{col}_encoded"
        train_df[train_encoded_col] = train_encoded
        
        # Apply to test data (if provided)
        if test_df is not None:
            test_encoded = te.transform(test_df[[col]])
            test_encoded_col = f"{col}_encoded"
            test_df[test_encoded_col] = test_encoded
            
        encoded_cols.append(train_encoded_col)

    # --- (Optional) Encode interaction columns ---
    if interactions:
        for col1, col2 in combinations(cat_cols, 2):
            # Construct an interaction feature in train
            train_interaction = train_df[col1].astype(str) + "_" + train_df[col2].astype(str)
            
            # We'll store it in a temporary column just for clarity
            train_df["_interaction"] = train_interaction
            
            # Fit a fresh TargetEncoder on this new "interaction" column
            te_inter = TargetEncoder(target_type="continuous")
            train_encoded = te_inter.fit_transform(train_df[["_interaction"]], train_df[target])
            
            # Create a column name for the interaction encoding
            interaction_encoded_col = f"{col1}_{col2}_encoded"
            train_df[interaction_encoded_col] = train_encoded
            
            # Encode the test data (if provided)
            if test_df is not None:
                test_interaction = test_df[col1].astype(str) + "_" + test_df[col2].astype(str)
                test_df["_interaction"] = test_interaction
                test_encoded = te_inter.transform(test_df[["_interaction"]])
                test_df[interaction_encoded_col] = test_encoded
                
                # Drop the temporary interaction column
                test_df.drop(columns="_interaction", inplace=True, errors="ignore")

            # Drop the temporary interaction column from train
            train_df.drop(columns="_interaction", inplace=True, errors="ignore")

            encoded_cols.append(interaction_encoded_col)

    return train_df, test_df, encoded_cols