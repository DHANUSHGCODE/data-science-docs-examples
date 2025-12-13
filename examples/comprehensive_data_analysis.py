#!/usr/bin/env python3
"""
Comprehensive Data Analysis Example
Demonstrates NumPy, Pandas, Matplotlib, and Scikit-learn integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    """Load and prepare dataset"""
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randn(100)
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess dataset"""
    scaler = StandardScaler()
    features = df.drop('target', axis=1)
    return scaler.fit_transform(features), df['target']

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models"""
    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
    
    return results

if __name__ == '__main__':
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    results = train_models(X_train, X_test, y_train, y_test)
    print('Model Results:', results)
