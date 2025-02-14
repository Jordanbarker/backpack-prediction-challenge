import numpy as np

from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

from model_config import l1_models, l2_models

import xgboost as xgb

def check_model_dicts(model_dict):
    """
    Checks that each model config has the required keys and a valid type.
    """
    required_keys = ['type', 'params', 'features']
    valid_model_types = ['xgb', 'lgb']

    for model_name, model_config in model_dict.items():
        # Check for required keys
        for key in required_keys:
            if key not in model_config:
                print(f"ERROR: Model '{model_name}' is missing required key '{key}'.")
                continue  # Skip further checks if a key is missing

        # Check if 'type' is valid
        if 'type' in model_config and model_config['type'] not in valid_model_types:
            print(f"ERROR: Model '{model_name}' has invalid type '{model_config['type']}'. "
                  f"Valid types: {valid_model_types}.")


def create_splits(train_df, te_cols, te_cols_renamed, target, n_splits=10, random_state=444):
    """
    Creates train/validation splits, applies target encoding and computes frequency maps.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data_splits = []

    for train_index, val_index in kf.split(train_df):
        train_fold = train_df.iloc[train_index].copy()
        val_fold = train_df.iloc[val_index].copy()

        # Apply target encoding
        te = TargetEncoder(target_type="continuous", smooth=20)
        train_fold[te_cols_renamed] = te.fit_transform(train_fold[te_cols], train_fold[target])
        val_fold[te_cols_renamed] = te.transform(val_fold[te_cols])

        # Compute frequency mapping for 'weight_capacity'
        frequency_map = train_fold['weight_capacity'].value_counts().to_dict()
        train_fold['weight_capacity_counts'] = train_fold['weight_capacity'].map(frequency_map)
        val_fold['weight_capacity_counts'] = val_fold['weight_capacity'].map(frequency_map).fillna(1)

        data_splits.append((train_fold, val_fold))

    return data_splits


def train_eval_cv(model_dict, data_splits, target):
    """
    Trains models defined in model_dict using cross-validation splits,
    and prints out the best and average RMSE.
    """
    check_model_dicts(model_dict)
    
    model_keys = list(model_dict.keys())
    rmse_list = []

    for i, (train_fold, val_fold) in enumerate(data_splits, 1):
        best_rmse = float('inf')

        for k, v in model_dict.items():
            params = v['params']
            features = v['features']
            
            if v['type'] == 'xgb':
                dtrain_fold = (
                    xgb.DMatrix(train_fold[features], 
                                label=train_fold[target], 
                                enable_categorical=True)
                )
                dvalid_fold = (
                    xgb.DMatrix(val_fold[features], 
                                label=val_fold[target], 
                                enable_categorical=True)
                )
                bst = xgb.train(
                    params=params,
                    dtrain=dtrain_fold,
                    evals=[(dtrain_fold, "train"), (dvalid_fold, "validation_0")],
                    verbose_eval=False,
                )
                val_fold[k] = bst.predict(dvalid_fold)
                rmse = root_mean_squared_error(val_fold[k], val_fold[target])
                best_rmse = rmse if rmse < best_rmse else best_rmse

        avg_pred = val_fold[model_keys].sum(axis=1) / len(model_keys)
        avg_rmse = root_mean_squared_error(avg_pred, val_fold[target])
        negative = "worse :(" if avg_rmse > best_rmse else ''
        print(f"[{i}] Best model: {best_rmse:.4f}, Avg: {avg_rmse:.4f}", negative)
        rmse_list.append(avg_rmse)

    print(f"Average: {np.mean(rmse_list):.4f}")


def main():
    # Check model dictionaries
    print("Checking L1 models:")
    check_model_dicts(l1_models)
    print("Checking L2 models:")
    check_model_dicts(l2_models)

    # Create cross-validation splits
    te_cols = ["weight_capacity"]            # Columns to target-encode
    te_cols_renamed = ["weight_capacity_te"] # Names for the encoded columns
    target = "your_target_column"

    # Example of how you'd get these splits
    data_splits = create_splits(train_df, te_cols, te_cols_renamed, target)

    # Train and evaluate L1 models
    print("=== Training L1 Models ===")
    train_eval_cv(l1_models, data_splits, target)

    # Optionally use L1 predictions to further add features for L2 or 
    # feed the same data_splits with appended columns back into L2.
    # Train and evaluate L2 models
    print("\n=== Training L2 Models ===")
    train_eval_cv(l2_models, data_splits, target)

if __name__ == "__main__":
    main()