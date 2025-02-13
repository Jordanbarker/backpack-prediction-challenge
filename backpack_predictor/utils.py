
import pandas as pd
from itertools import combinations

from .features import target

from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder


def prepare_data(df: pd.DataFrame, is_train: bool = True):
    """
    Prepares the dataset for training or testing by renaming columns, handling missing values,
    converting categorical and numerical features.
    
    Args:
        df (pd.DataFrame): The input dataframe (train or test).
        is_train (bool): Indicates if the dataframe is training data (default is True).
        
    Returns:
        pd.DataFrame: The processed dataframe.
    """
    
    # Rename columns
    columns = [
        'id', 'brand', 'material', 'size', 'compartments', 
        'laptop_compartment', 'is_waterproof', 'style', 'color', 
        'weight_capacity'
    ]
    
    if is_train:
        columns.append('price')
    
    df.columns = columns
    
    if is_train:
        # Keep the id column on the test set since it's used for the submission
        df = df.drop(columns='id')

    # Convert categories to int
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    df["size"] = df["size"].map(size_mapping).fillna(-1).astype(int)

    brand_mapping = {"Adidas": 0, "Puma": 1, "Nike": 2, "Jansport": 3, "Under Armour": 4}
    df["brand"] = df["brand"].map(brand_mapping).fillna(-1).astype(int)

    material_mapping = {"Leather": 0, "Nylon": 1, "Polyester": 2, "Canvas": 3}
    df["material"] = df["material"].map(material_mapping).fillna(-1).astype(int)

    style_mapping = {"Backpack": 0, "Messenger": 1, "Tote": 2}
    df["style"] = df["style"].map(style_mapping).fillna(-1).astype(int)

    color_mapping = {"Black": 0, "Gray": 1, "Red": 2, "Pink": 3, "Green": 4, "Blue": 5}
    df["color"] = df["color"].map(color_mapping).fillna(-1).astype(int)

    binary_mapping = {"No": 0, "Yes": 1}
    df["laptop_compartment"] = df["laptop_compartment"].map(binary_mapping).fillna(-1).astype(int)
    df["is_waterproof"] = df["is_waterproof"].map(binary_mapping).fillna(-1).astype(int)
    
    df['weight_capacity'] = df['weight_capacity'].fillna(-1)
    df['compartments'] = df['compartments'].astype(int)
    
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