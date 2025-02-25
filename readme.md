#  Backpack Price Prediction - Kaggle Playground (Season 5, Episode 2)

This repository contains my code for the Backpack Prediction Challenge from the Kaggle Playground Series (2025). The goal is to predict backpack prices based on various attributes using machine learning techniques.

## Approach
1. Data Exploration: Initial EDA to understand feature distributions and correlations.
2. Feature Engineering: Created additional features to enhance model performance.
3. Modeling: Used various algorithms, including ensemble methods and hyperparameter tuning. Optuna was used for hyperparameter tuning. Files can be viewed like so:

## Hyperparam testing
This is done with optuna and can be monitored like so:
- optuna-dashboard sqlite:///db.sqlite3
- optuna-dashboard sqlite:///lgb_db.sqlite3