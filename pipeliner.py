import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer

from research_preprocessors import (PreprocessorBasic,
                                    PreprocessorRelativePower,
                                    PreprocessorYearSquared,
                                    PreprocessorTFIDF)

def run_pipe_basic(X_train, y_train, fill_method='median'):
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols)
        ], remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorBasic(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {
        'model__alpha': [0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='r2',
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    print('*'*50)

def run_pipe_relative_power(X_train, y_train, fill_method='median'):
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm',
                'hppl', 'nmpl']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols)
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorRelativePower(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {
        'model__alpha': [0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='r2',
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    print('*'*50)

def run_pipe_year_squared(X_train, y_train, fill_method='median'):
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm',
                'hppl', 'nmpl', 'year_squared']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols)
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorYearSquared(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {
        'model__alpha': [0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='r2',
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    print('*'*50)

def run_pipe_tfidf(X_train, y_train, fill_method='median'):
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm',
                'hppl', 'nmpl', 'year_squared']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols),
            ('text', text_transformer, 'name')
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorTFIDF(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {
        'model__alpha': [0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='r2',
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    print('*'*50)

def run_pipe_business_ridge(X_train, y_train, fill_method='median'):
    # Определяем новый scorer
    def business_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        errors = np.abs((y_true - y_pred) / y_true)
        result = np.mean(errors > 0.1)

        return result

    from sklearn.metrics import make_scorer

    business_scorer = make_scorer(business_metrics, greater_is_better=False)
    
    # Дальше все как обычно, только указываем новый метод в scoring
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm',
                'hppl', 'nmpl', 'year_squared']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols),
            ('text', text_transformer, 'name')
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorTFIDF(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {
        'model__alpha': [0, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring=business_scorer,
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best businness Score:", grid_search.best_score_ * (-1))
    print('*'*50)

def run_pipe_business_lasso(X_train, y_train, fill_method='median'):
    # Определяем новый scorer
    def business_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        errors = np.abs((y_true - y_pred) / y_true)
        result = np.mean(errors > 0.1)

        return result

    from sklearn.metrics import make_scorer

    business_scorer = make_scorer(business_metrics, greater_is_better=False)
    
    # Дальше все как обычно, только указываем новый метод в scoring
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm',
                'hppl', 'nmpl', 'year_squared']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols),
            ('text', text_transformer, 'name')
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorTFIDF(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Lasso())
    ])

    param_grid = {
        'model__alpha': [0, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring=business_scorer,
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best businness Score:", grid_search.best_score_ * (-1))
    print('*'*50)

def run_pipe_ridge_weighted_mse(X_train, y_train, fill_method='median'):
    # Определяем новый scorer
    def weghted_mse(y_true, y_pred, coef = 2):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        errors = np.where(
            (y_pred - y_true) < 0,
            coef * (y_pred - y_true)**2,
            (y_pred - y_true)**2
        )
        result = np.mean(errors)

        return result

    business_scorer = make_scorer(weghted_mse, greater_is_better=False)
    
    # Дальше все как обычно, только указываем новый метод в scoring
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm',
                'hppl', 'nmpl', 'year_squared']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols),
            ('text', text_transformer, 'name')
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorTFIDF(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {
        'model__alpha': [0, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring=business_scorer,
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best weighted MSE:", grid_search.best_score_ * (-1))
    print("Best weighted RMSE:", np.sqrt(grid_search.best_score_ * (-1)))
    print('*'*50)

def run_pipe_lasso_weighted_mse(X_train, y_train, fill_method='median'):
    # Определяем новый scorer
    def weghted_mse(y_true, y_pred, coef = 2):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        errors = np.where(
            (y_pred - y_true) < 0,
            coef * (y_pred - y_true)**2,
            (y_pred - y_true)**2
        )
        result = np.mean(errors)

        return result

    business_scorer = make_scorer(weghted_mse, greater_is_better=False)
    
    # Дальше все как обычно, только указываем новый метод в scoring
    number_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                'max_torque', 'min_torque_rpm', 'max_torque_rpm',
                'hppl', 'nmpl', 'year_squared']

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'seats']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, number_cols),
            ('cat', cat_transformer, cat_cols),
            ('text', text_transformer, 'name')
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('super_preprocessor', PreprocessorTFIDF(fill_method=fill_method)),
        ('preprocessor', preprocessor),
        ('model', Lasso())
    ])

    param_grid = {
        'model__alpha': [0, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring=business_scorer,
        verbose=4,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print('*'*50)
    print(grid_search.best_params_)
    print("Best weighted MSE:", grid_search.best_score_ * (-1))
    print("Best weighted RMSE:", np.sqrt(grid_search.best_score_ * (-1)))
    print('*'*50)

