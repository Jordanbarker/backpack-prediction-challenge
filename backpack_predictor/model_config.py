features_to_TE = ['weight_capacity', 'compartments', 'wc_decimal_count', 'laptop_compartment', 'is_waterproof', 'size', 'material', 'style']
features_te = [col + "_te" for col in features_to_TE]
features = features_te + ['weight_capacity', 'compartments']

######################
# XGBoost
###################### 
common_xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}

# xgb_params_0 = {'colsample_bylevel': 0.6749490938559761, 'colsample_bytree': 0.9790694501766262, 'learning_rate': 0.11559290622818262, 'max_depth': 6, 'min_child_weight': 12, 'gamma': 0.8382681285208085, 'subsample': 0.5874192575935778, 'reg_alpha': 3.3622771363878876e-05, 'reg_lambda': 3.141420086780599e-06, "early_stopping_rounds": 20, "num_boost_round": 200,}
# xgb_params_1 = {'colsample_bylevel': 0.5210471720222337, 'colsample_bytree': 0.9558753657785292, 'learning_rate': 0.21268822566819126, 'max_depth': 6, 'min_child_weight': 30, 'gamma': 0.7323111677091081, 'subsample': 0.7482950178956017, 'reg_alpha': 6.266062113483953e-05, 'reg_lambda': 0.15641348826123197, "early_stopping_rounds": 20, "num_boost_round": 200,}

# 38.66
xgb_params_2 = {'colsample_bylevel': 0.5855818440243894, 'colsample_bytree': 0.6608307328204897, 'learning_rate': 0.004490371142580223, 'max_depth': 8, 'min_child_weight': 14, 'gamma': 0.6076023861910245, 'subsample': 0.2778865610345461, 'reg_alpha': 0.033705063323414546, 'reg_lambda': 0.00010482019100779534, 'early_stopping_rounds': 420, 'num_boost_round': 3739}

# 38.661
xgb_params_3 = {'colsample_bylevel': 0.9387587145081213, 'colsample_bytree': 0.6506691909264717, 'learning_rate': 0.0850463496255034, 'max_depth': 6, 'min_child_weight': 35, 'gamma': 0.6729529635650856, 'subsample': 0.7799353562727942, 'reg_alpha': 0.0007769752893030054, 'reg_lambda': 0.00013696944943227977}

######################
# Catboost
######################
common_catboost_params = {
    'verbose': 0,
    'eval_metric': 'RMSE'
}
# 38.66765 with 20 early stopping rounds
catboost_params_0 = {'learning_rate': 0.06531304678858534, 'iterations': 3649, 'depth': 14, 'l2_leaf_reg': 3.5480233728676995, 'random_strength': 22.013121863975286, 'bootstrap_type': 'Bayesian', 'boosting_type': 'Plain', 'colsample_bylevel': 0.09689855613070206, 'bagging_temperature': 0.3520912621795899, 'random_state': 42, }

# 38.673
catboost_params_1 = {'learning_rate': 0.12794147329741268, 'iterations': 1860, 'depth': 15, 'l2_leaf_reg': 7.87447241512, 'random_strength': 27.42065099934876, 'bootstrap_type': 'Bernoulli', 'boosting_type': 'Ordered', 'colsample_bylevel': 0.09919733865407444, 'subsample': 0.17380922614986466}

# 38.672
catboost_params_2 = {'learning_rate': 0.24665670782486956, 'iterations': 4469, 'depth': 12, 'l2_leaf_reg': 5.788398385844328, 'random_strength': 35.169068482278604, 'bootstrap_type': 'MVS', 'boosting_type': 'Plain', 'colsample_bylevel': 0.04811465917313007, 'early_stopping_rounds': 245}

# 38.665
catboost_params_3 = {'learning_rate': 0.048978640898339074, 'iterations': 3018, 'depth': 12, 'l2_leaf_reg': 7.6279955551693455, 'random_strength': 15.344154303793191, 'bootstrap_type': 'Bayesian', 'boosting_type': 'Ordered', 'colsample_bylevel': 0.09578693952226389, 'bagging_temperature': 0.3824045323195606, 'early_stopping_rounds': 975, 'random_state': 42, 'verbose': 0}

######################
# LightGBM
######################    
common_lgb_params = {
    'verbose': -1,
    'objective': 'regression',
    'metric': 'rmse',
    'force_row_wise': True,
}

# 38.655
lgb_params_0 = {'early_stopping_rounds': 390, 'bagging_fraction': 0.9954892076290114, 'bagging_freq': 3, 'cat_l2': 19.943225836972967, 'extra_trees': False, 'feature_fraction': 0.9583014726425529, 'learning_rate': 0.21124741547142464, 'max_bin': 7566, 'max_depth': 648, 'min_samples_leaf': 14, 'n_estimators': 1570, 'num_leaves': 68, 'lambda_l1': 7.772617123196223e-06, 'lambda_l2': 0.0006810978623249646}

# 38.653 - converted cols to cats, dropped 'laptop_compartment', 'is_waterproof'
lgb_params_1 = {'early_stopping_rounds': 916, 'cat_l2': 40.57932515907615, 'extra_trees': False, 'feature_fraction': 0.9028265875188044, 'learning_rate': 0.02133798565133846, 'max_bin': 7449, 'max_depth': 590, 'min_samples_leaf': 22, 'n_estimators': 14706, 'num_leaves': 141, 'lambda_l1': 0.004931837373939155, 'lambda_l2': 2.1647829905701264e-06}

######################
# Level 1
###################### 
l1_models = {
    'xgb_l1_0': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_2, "seed": 42},
        'features': features
    },
    'xgb_l1_1': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_3, "seed": 44},
        'features': features
    },
    'catboost_l1_0': {
        'type': 'catboost',
        'params': {**common_catboost_params, **catboost_params_0},
        'features': features
    },
    'catboost_l1_1': {
        'type': 'catboost',
        'params': {**common_catboost_params, **catboost_params_1},
        'features': features
    },
    'lgb_l1_0': {
        'type': 'lgb',
        'params': {**common_lgb_params, **lgb_params_0},
        'features': features
    },
    'lgb_l1_1': {
        'type': 'lgb',
        'params': {**common_lgb_params, **lgb_params_1},
        'features': features
    },
}
l1_pred_cols = list(l1_models.keys())

######################
# Level 2
###################### 

l2_models = {
    'xgb_l2_0': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_2, "seed": 42},
        'features': features + l1_pred_cols
    },
    'xgb_l2_1': {
        'type': 'xgb',
        'params': {**common_xgb_params, **xgb_params_3, "seed": 44},
        'features': features + l1_pred_cols
    },
    'catboost_l2_0': {
        'type': 'catboost',
        'params': {**common_catboost_params, **catboost_params_0},
        'features': features + l1_pred_cols
    },
    'catboost_l2_1': {
        'type': 'catboost',
        'params': {**common_catboost_params, **catboost_params_1},
        'features': features + l1_pred_cols
    },
    'lgb_l2_0': {
        'type': 'lgb',
        'params': {**common_lgb_params, **lgb_params_0},
        'features': features + l1_pred_cols
    },
    'lgb_l2_1': {
        'type': 'lgb',
        'params': {**common_lgb_params, **lgb_params_1},
        'features': features + l1_pred_cols
    },
}
l2_pred_cols = list(l2_models.keys())

######################
# Level 3
###################### 

l3_models = {
    # 'elastic_net_l3': {
    #     'type': 'elastic_net',
    #     'params': {"seed": 42},
    #     'features': ["weight_capacity_te"] + l1_pred_cols + l2_pred_cols
    # },
    'lgb_l3_0': {
        'type': 'lgb',
        'params': {**common_lgb_params, **lgb_params_0},
        'features': features + l1_pred_cols + l2_pred_cols
    },
    'lgb_l3_1': {
        'type': 'lgb',
        'params': {**common_lgb_params, **lgb_params_0},
        'features': features + l1_pred_cols + l2_pred_cols
    },
}

######################
# Packaged Model Definition
###################### 
model_dict = {
    'models': [l1_models, l2_models, l3_models],
    'features_to_TE': features_to_TE,
    'features_te': features_te,
    'features': features,
}
