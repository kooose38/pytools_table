from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS
import pandas as pd 
from typing import Union 
import plotly.express as px 
import numpy as np 
import matplotlib.pyplot as plt


class LDA:
  def __init__(self):
    self.model = None 

  def fit(self, n_components: int=1):
    self.model = LinearDiscriminantAnalysis(n_components=n_components)

  def transform(self, x_val: pd.DataFrame) -> np.ndarray:
    x_val_c = x_val.copy()
    return self.model.transform(x_val_c)
    

class TSNE_:
  def __init__(self):
    self.model = None 

  def fit(self, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series], 
          n_components: int=2, perplexity: int=25, random_state: int=0, is_plot: bool=False):
    self.model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit(x_train)
    if (is_plot) and (n_components == 2):
      self._plot(x_train, y_train)

  def _plot(self, x_train, y_train):
    x_t = self.model.fit_transform(x_train)
    fig = px.scatter(x=x_t[:, 0], y=x_t[:, 1], color=y_train, title="TSNE:")
    fig.show()

  def transform(self, x: pd.DataFrame) -> np.ndarray:
    return self.model.fit_transform(x)
    
    
class MDS_:
  def __init__(self):
    self.model = None 

  def fit(self, x_train: pd.DataFrame, n_components: int=2, metric: bool=True, random_state: int=0, 
          is_plot: bool=False):
    self.model = MDS(n_components=n_components, metric=metric, random_state=random_state)

    if is_plot:
      self._plot(x_train)

  def _plot(self, x_train):
    l = []
    for i in np.arange(len(x_train.columns)):
      model = MDS(n_components=i+1, metric=True, random_state=0)
      model.fit_transform(x_train)
      l.append(model.stress_)

    plt.plot(np.arange(len(x_train.columns))+1, l)
    plt.xlabel("number of dimensions")
    plt.ylabel("stress")
    plt.xlim(0, len(x_train.columns))
    plt.xticks(np.arange(len(x_train.columns))+1)
    plt.grid()
    plt.show()

  def transform(self, x: pd.DataFrame) -> np.ndarray:
    return self.model.fit_transform(x)


class PCA_:
  def __init__(self):
    self.model = None 

  def fit(self, x_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame], 
          n_components: int=2, is_plot: bool=False) -> np.ndarray:
    self.model = PCA(n_components=n_components).fit(x_train)

    if is_plot and n_components == 2:
      self._plot(x_train, y_train)

    return self.model.explained_variance_ratio_

  def _plot(self, x_train, y_train):
    x_pca = self.model.fit_transform(x_train)
    fig = px.scatter(x=x_pca[:, 0], y=x_pca[:, 1], color=y_train, title="PCA:")
    fig.show()

  def transform(self, x: pd.DataFrame) -> np.ndarray:
    return self.model.transform(x)
