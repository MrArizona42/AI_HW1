import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SuperPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                fill_method='median',
                scalar=0
            ) -> None:
        
        assert fill_method in ['median', 'mean', 'scalar'], f"There's no such filling method: {fill_method}"
        self.fill_method = fill_method

        self.scalar = scalar

    def fit(self, X, y=None):
        X_transformed = X.copy()

        X_transformed = self.convert_columns(X_transformed)

        X_transformed = self.get_brand_model(X_transformed)

        self.fill_vals={}

        if self.fill_method == 'median':
            self.fill_vals = X_transformed.select_dtypes('number').median().to_dict()

        elif self.fill_method == 'mean':
            self.fill_vals = X_transformed.select_dtypes('number').mean().to_dict()

        elif self.fill_method == 'scalar':
            for col in X_transformed.select_dtypes('number').columns:
                self.fill_vals[col] = self.scalar

        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()

        X_transformed = self.convert_columns(X_transformed)

        X_transformed = self.get_brand_model(X_transformed)

        X_transformed = self.fill_nans(X_transformed)

        return X_transformed
    
    def convert_columns(self, X):
        # Тут просто нужно выделить числа из строк
        pattern = r'(\d+\.?\d+?)'
        X['mileage'] = X['mileage'].str.extract(pattern).astype('float')
        X['engine'] = X['engine'].str.extract(pattern).astype('float')
        X['max_power'] = X['max_power'].str.extract(pattern).astype('float')

        # Разделяем torque на 2 колонки
        X[['max_torque', 'torque_rpm']] = X['torque'].str.split('@|at', expand=True, n=1)
        X = X.drop('torque', axis=1)

        # Определяем единицы измерения момента
        X['torque_measure'] = np.where(
            X['max_torque'].str.lower().str.extract('(kgm|nm)', expand=False).isna(),
            X['torque_rpm'].str.lower().str.extract('(kgm|nm)', expand=False),
            X['max_torque'].str.lower().str.extract('(kgm|nm)', expand=False)
        )

        # Определяем максимальный момент и переводим в нужные единицы если нужно
        pattern = r'(\d+\.?\d+?)'
        X['max_torque'] = X['max_torque'].str.extract(pattern).astype('float')
        X['max_torque'] = np.where(
            X['torque_measure'] == 'kgm',
            X['max_torque']*9.8,
            X['max_torque']
        )
        X = X.drop('torque_measure', axis=1)

        # Выделяем диапазон оборотов максимального момента
        pattern = r'(\d+\,?\d+)-?(\d+\,?\d+)?'
        X[['min_torque_rpm', 'max_torque_rpm']] = X['torque_rpm'].str.extract(pattern)
        X = X.drop('torque_rpm', axis=1)

        X['min_torque_rpm'] = X['min_torque_rpm'].str.replace(',', '').astype('float')
        X['max_torque_rpm'] = X['max_torque_rpm'].str.replace(',', '').astype('float')
        X['max_torque_rpm'] = np.where(
            X['max_torque_rpm'].isna(),
            X['min_torque_rpm'],
            X['max_torque_rpm']
        )

        X['hppl'] = X['max_power'] / X['engine'] * 1000
        X['nmpl'] = X['max_torque'] / X['engine'] * 1000

        X['year_squared'] = X['year'] * X['year']
        X['year_squared'] = X['year'] * X['year']

        # X = pd.merge(
        #     left=X,
        #     right=X.isna().astype('int'),
        #     left_index=True,
        #     right_index=True,
        #     suffixes=['', '_isna']
        # )

        return X
    
    def fill_nans(self, X):
        for col, val in self.fill_vals.items():
            X[col] = X[col].fillna(val)

        return X
    
    def get_brand_model(self, X):
        X[['brand', 'model']] = X['name'].str.split(' ', expand=True)[[0, 1]]
        X['name'] = X['name'].str.split(expand=True, n=2)[2]

        return X