import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union 

class TargetEncoder(BaseEstimator, TransformerMixin):
  '''訓練データのラベルを使ってターゲットエンコーダーを行う'''
  def __init__(self):
    self.x_train = None  
    self.y_train = None  
    self.col = ""
    self.sample = None 
    self._data_tmp = None 

  def fit(self, x_train, y_train, col: str, method: Union["mean", "median"]="mean"):
    self.x_train = x_train 
    self.y_train = y_train 
    self.col = col 
    self._data_tmp = pd.DataFrame({col: self.x_train[col], "target": self.y_train.values.ravel()})
    if method == "mean":
      self.sample = self._data_tmp.groupby(col)["target"].mean()
    elif method == "median":
      self.sample = self._data_tmp.groupby(col)["target"].median()
    return self.sample 

  def transform(self, x_test, n_splits: int=4):
    test = x_test.copy()
    test[self.col] = test[self.col].map(self.sample)

    dummy = np.repeat(np.nan, self.x_train.shape[0])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for id1, id2 in kf.split(self.x_train):
      sample = self._data_tmp.iloc[id1].groupby(self.col)["target"].mean()
      dummy[id2] = self.x_train[self.col].iloc[id2].map(sample)
    train = self.x_train.copy() 
    train[self.col] = dummy 
    return train, test 


class FrequenceEncoder(BaseEstimator, TransformerMixin):
  '''カテゴリーの出現頻度に応じて数値変換する'''
  def __init__(self):
    freq = None 
    col = ""

  def fit(self, x_train, col: str):
    self.col = col 
    self.freq = x_train[col].value_counts()

  def tranform(self, x_test):
    test = x_test.copy()
    return test[self.col].map(freq)

