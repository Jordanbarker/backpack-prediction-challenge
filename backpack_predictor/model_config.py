common_xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
    "num_boost_round": 200,
    "early_stopping_rounds": 20,
}

xgb_params_0 = {'colsample_bylevel': 0.6749490938559761, 'colsample_bytree': 0.9790694501766262, 'learning_rate': 0.11559290622818262, 'max_depth': 6, 'min_child_weight': 12, 'gamma': 0.8382681285208085, 'subsample': 0.5874192575935778, 'reg_alpha': 3.3622771363878876e-05, 'reg_lambda': 3.141420086780599e-06}
xgb_params_1 = {'colsample_bylevel': 0.5210471720222337, 'colsample_bytree': 0.9558753657785292, 'learning_rate': 0.21268822566819126, 'max_depth': 6, 'min_child_weight': 30, 'gamma': 0.7323111677091081, 'subsample': 0.7482950178956017, 'reg_alpha': 6.266062113483953e-05, 'reg_lambda': 0.15641348826123197}

# 38.66
xgb_params_2 = {'colsample_bylevel': 0.5855818440243894, 'colsample_bytree': 0.6608307328204897, 'learning_rate': 0.004490371142580223, 'max_depth': 8, 'min_child_weight': 14, 'gamma': 0.6076023861910245, 'subsample': 0.2778865610345461, 'reg_alpha': 0.033705063323414546, 'reg_lambda': 0.00010482019100779534, 'early_stopping_rounds': 420, 'num_boost_round': 3739}

# 38.661
xgb_params_3 = {'colsample_bylevel': 0.9387587145081213, 'colsample_bytree': 0.6506691909264717, 'learning_rate': 0.0850463496255034, 'max_depth': 6, 'min_child_weight': 35, 'gamma': 0.6729529635650856, 'subsample': 0.7799353562727942, 'reg_alpha': 0.0007769752893030054, 'reg_lambda': 0.00013696944943227977}

# 38.66765 with 20 early stopping rounds
catboost_params_0 = {
    'learning_rate': 0.06531304678858534,
    'iterations': 3649,
    'depth': 14,
    'l2_leaf_reg': 3.5480233728676995,
    'random_strength': 22.013121863975286,
    'bootstrap_type': 'Bayesian',
    'boosting_type': 'Plain',
    'colsample_bylevel': 0.09689855613070206,
    'bagging_temperature': 0.3520912621795899,
    'random_state': 42,
    'verbose': 0,
    'eval_metric': 'RMSE'
}

# 38.673
catboost_params_1 = {'learning_rate': 0.12794147329741268, 'iterations': 1860, 'depth': 15, 'l2_leaf_reg': 7.87447241512, 'random_strength': 27.42065099934876, 'bootstrap_type': 'Bernoulli', 'boosting_type': 'Ordered', 'colsample_bylevel': 0.09919733865407444, 'subsample': 0.17380922614986466}

l1_models = {
    'xgb_l1_0': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_0, "seed": 42},
        'features': ["weight_capacity_te", "weight_capacity_counts"]
    },
    'xgb_l1_1': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_1, "seed": 44},
        'features': ["weight_capacity_te", "weight_capacity_counts", "compartments"]
    },
}

l1_pred_cols = list(l1_models.keys())

l2_models = {
    'xgb_l2_0': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_0, "seed": 42},
        'features': ["weight_capacity_te", "weight_capacity_counts"] + l1_pred_cols
    },
    'xgb_l2_1': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_1, "seed": 44},
        'features': ["weight_capacity_te", "weight_capacity_counts", "compartments"] + l1_pred_cols
    },
}

l2_pred_cols = list(l2_models.keys())

l3_models = {
    'elastic_net_l3': {
        'type': 'elastic_net',
        'params': {"seed": 42},
        'features': ["weight_capacity_te"] + l1_pred_cols + l2_pred_cols
    },
}