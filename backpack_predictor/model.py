import numpy as np
import pandas as pd 
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import ElasticNetCV

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from loguru import logger

def check_model_dicts(model_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Checks that each model config has the required keys and a valid type.
    """
    if not isinstance(model_dict, dict):
        logger.error("model_dict must be a dictionary with string keys and dictionary values.")
        raise TypeError("ERROR: model_dict must be a dictionary with string keys and dictionary values.")

    required_keys = ['type', 'params', 'features']
    valid_model_types = ['xgb', 'lgb', 'catboost', 'elastic_net']

    for model_name, model_config in model_dict.items():
        if not isinstance(model_name, str):
            raise TypeError(f"ERROR: Model name must be a string. Got: '{model_name}'")
        
        if not isinstance(model_config, dict):
            raise TypeError(f"ERROR: Model config must be a dict. Got: '{model_config}'")

        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise KeyError(f"Missing required keys {missing_keys} in model '{model_name}'.")

        # Check if 'type' is valid
        if 'type' in model_config and model_config['type'] not in valid_model_types:
            raise ValueError(
                f"Invalid type '{model_config['type']}' for model '{model_name}'. "
                f"Expected one of {valid_model_types}."
            )
        
    logger.info(f'Passed checks for model_dict: {model_dict}')


def create_splits(
    train_df: pd.DataFrame, 
    te_cols: List[str], 
    te_cols_renamed: List[str], 
    target: str, 
    n_splits: int = 10, 
    random_state: int = 444
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Creates train/validation splits, applies target encoding and computes frequency maps.
    """
    logger.info(f"Creating {n_splits} CV splits.")

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


def train_eval_cv(model_dict: Dict[str, Dict[str, Any]],
                  data_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
                  target: str) -> pd.DataFrame:
    """
    Trains models defined in model_dict using cross-validation splits,
    logs out performance, and returns the concatenated validation folds
    with model predictions included.

    :param model_dict: dictionary specifying models, their parameters, and features
    :param data_splits: list of (train_fold, val_fold) pairs
    :param target: name of the target column
    :return: DataFrame containing validation folds with predictions
    """
    check_model_dicts(model_dict)
    
    model_keys = list(model_dict.keys())
    rmse_list = []

    logger.info(f'Starting train / eval CV loop for model_dict: {model_dict}')
    for i, (train_fold, val_fold) in enumerate(data_splits, 1):
        best_rmse = float('inf')

        for model_name, model_conf in model_dict.items():
            model_type = model_conf['type']
            params = model_conf['params']
            features = model_conf['features']
            
            ######################
            # XGBoost
            ######################            
            if model_type == 'xgb':
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
                model = xgb.train(
                    params=params,
                    dtrain=dtrain_fold,
                    evals=[(dtrain_fold, "train"), (dvalid_fold, "validation_0")],
                    verbose_eval=False,
                )
                val_fold[model_name] = model.predict(dvalid_fold)

            ######################
            # LightGBM
            ######################    
            elif model_type == 'lgb':
                train_data = lgb.Dataset(train_fold[features], label=train_fold[target])
                valid_data = lgb.Dataset(val_fold[features], label=val_fold[target], reference=train_data)
            
                model = lgb.train(
                    params=params,
                    train_set=train_data,
                    valid_sets=[train_data, valid_data],
                    valid_names=['train_0', 'valid_0'],
                )
                val_fold[model_name] = model.predict(val_fold[features], num_iteration=model.best_iteration)

            ######################
            # Catboost
            ######################
            elif model_type == 'catboost':
                model = CatBoostRegressor(**params)
                model.fit(
                    train_fold[features], 
                    train_fold[target],
                    eval_set=[(val_fold[features], val_fold[target])],
                    early_stopping_rounds=50, # TODO 
                    use_best_model=True
                )
                val_fold[model_name] = model.predict(val_fold[features])

            ######################
            # ElasticNet
            ######################
            elif model_type == 'elastic_net':
                clf = ElasticNetCV(
                    alphas=[1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0],
                    l1_ratio=np.arange(0.01, 1, 0.01),
                    cv=5,
                    n_jobs=-1,
                ).fit(train_fold[features], train_fold[target])
                val_fold[model_name] = clf.predict(val_fold[features])

                logger.debug(f"[{i}] ElasticNet alpha: {clf.alpha_}")
                logger.debug(f"[{i}] ElasticNet l1 ratio: {clf.l1_ratio_}")
                logger.debug(f"[{i}] ElasticNet coefs: {clf.coef_}")
            else:
                raise ValueError(f"Model type '{model_type}' is not implemented in the training loop.")

            rmse = root_mean_squared_error(val_fold[model_name], val_fold[target])
            logger.debug(f"[{i}] RMSE: {rmse:.3f} ({model_name})")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name


        avg_pred = val_fold[model_keys].mean(axis=1)
        avg_rmse = root_mean_squared_error(avg_pred, val_fold[target])
        if avg_rmse > best_rmse:
            logger.debug(f"[{i}] Single model outperformed the average.")

        logger.info(f"[{i}] Average RMSE: {avg_rmse:.4f} Best model ({best_model}): {best_rmse:.4f}")
        rmse_list.append(avg_rmse)

    logger.info(f"Average RMSE across all folds: {np.mean(rmse_list):.4f}")

    next_level_df = pd.concat([split[1] for split in data_splits])
    return next_level_df