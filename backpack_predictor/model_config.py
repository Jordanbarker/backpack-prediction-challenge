common_xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
    "num_boost_round": 200,
    "early_stopping_rounds": 20,
}

l1_models = {
    'xgb_l1_0': {
        'type': 'xgb',
        'params': {**common_xgb_params, "seed": 42},
        'features': ["weight_capacity_te", "weight_capacity_counts"]
    },
    'xgb_l1_1': {
        'type': 'xgb',
        'params': {**common_xgb_params, "seed": 44},
        'features': ["weight_capacity_te", "weight_capacity_counts", "compartments"]
    },
}

l1_pred_cols = list(l1_models.keys())
l2_models = {
    'xgb_l2_0': {
        'type': 'xgb',
        'params': {**common_xgb_params, "seed": 42},
        'features': ["weight_capacity_te", "weight_capacity_counts"] + l1_pred_cols
    },
    'xgb_l2_1': {
        'type': 'xgb',
        'params': {**common_xgb_params, "seed": 44},
        'features': ["weight_capacity_te", "weight_capacity_counts", "compartments"] + l1_pred_cols
    },
}