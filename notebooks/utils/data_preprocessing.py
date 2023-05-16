import pandas as pd
from datetime import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline

class ChangeDtypes(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._feature_names = ['first_issue_date', 'first_redeem_date', 'last_transaction_datetime']
        #print(self._feature_names)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self._feature_names] = X_[self._feature_names].apply(pd.to_datetime)
        if 'last_transaction_datetime' in self._feature_names:
             X_['last_transaction_datetime'] = X_['last_transaction_datetime'].apply(lambda x: x.replace(tzinfo=None))
        return X_
    

class DataTreatment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_age = None
        return None
    
    def fit(self, X, y=None):
        self.median_age = X['age'].median()
        return self
    
    def transform(self, X, y=None):
        max_first_redeem = X['first_redeem_date'].max() + pd.Timedelta(days=365)
        X_ = X.copy()
        X_.loc[(X_['age']<10) | (X_['age'] >100), 'age'] = self.median_age
        X_['first_redeem_date'] = X_['first_redeem_date'].fillna(max_first_redeem)
        return X_
    

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy()
        X_['recency'] = X_['last_transaction_datetime'].apply(lambda x: (pd.Timestamp('2019-03-20')-x).days)
        X_['avg_ticket'] = X_[['total_amount_spent', 'n_transactions']].apply(lambda x: x[0]/x[1], axis=1)
        X_['issue_redeem_delay'] = X_[['first_issue_date', 'first_redeem_date']].apply(lambda x: (x[1]-x[0]).days, axis=1)

        X_ = X_.drop(columns=['first_issue_date', 'first_redeem_date', 'last_transaction_datetime', 'client_id'])

        return X_
    