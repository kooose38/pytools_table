from typing import Union, Any 
import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

class Logarithmic(BaseEstimator, TransformerMixin):
  '''対数変換'''
  def __init__(self):
    pass 

  def __call__(self, series: pd.Series) -> np.ndarray:
    x = series.values
    x = np.sign(x) * np.log(np.abs(x))
    return x 

class Logarithmic1p(BaseEstimator, TransformerMixin):
  '''負の値を含む対数変換'''
  def __init__(self):
    pass 

  def __call__(self, series: pd.Series) -> pd.Series:
    x = series.values
    x = np.log1p(x)
    return x 

class OrderScale(BaseEstimator, TransformerMixin):
  '''順序尺度変換'''
  def __init__(self):
    pass 

  def __call__(self, series: pd.Series, n_bins: int=4, labels: bool=False) -> pd.Series:
     return pd.cut(series, n_bins, labels=labels)

class NormDist(BaseEstimator, TransformerMixin):
  '''正規分布変換'''
  def __init__(self, method: Union["box-cox", "yao-johnson"]="box-cox"):
    self.method = method 
    self.model = None 
    self.col = ""

  def fit(self, series: pd.Series):
    self.model = PowerTransformer(method=self.method)
    self.model.fit(series)
    self.col = series.name 

  def transform(self, series: pd.Series) -> Union[pd.Series, Any]:
    if self.model is not None:
      return self.model.transform(series)
    else:
      raise NotImplementedError
