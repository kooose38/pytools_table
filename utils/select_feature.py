
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs 
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs 
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd 
from typing import List, Any, Union
import matplotlib.pyplot as plt 
import numpy as np 

class SelectKBest_:
  def __doc__(self):
     '''
     統計量から特長量抽出をする。
     主に分類タスクには、カイ２乗やF値を用いる。
     '''

  def __init__(self):
    self.model_type = ""

  def fit(self, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series],
          x_val: pd.DataFrame, y_val: Union[pd.Series, pd.DataFrame], model_type: Union["classifier", "regression"]="classifier") -> List[str]:
    self.model_type = model_type 
    train_score, val_score, features = [], [], []
    for i in range(len(x_train.columns)):
      if model_type == "classifier":
        sel = SelectKBest(f_classif, k=i+1).fit(x_train, y_train)
      else:
        sel = SelectKBest(f_regression, k=i+1).fit(x_train, y_train)
      x_train_sel, x_val_sel = sel.transform(x_train), sel.transform(x_val)
      model = LogisticRegression().fit(x_train_sel, y_train)
      if model_type == "classifier":
        train_score.append(accuracy_score(y_train, model.predict(x_train_sel)))
        val_score.append(accuracy_score(y_val, model.predict(x_val_sel)))
      elif model_type == "regression":
        train_score.append(mean_squared_error(y_val, model.predict(x_train_sel)*(-1)))
        train_score.append(mean_squared_error(y_val, model.predict(x_val_sel)*(-1)))
      features.append(x_train.columns[sel.get_support()])

    self._plot(train_score, val_score, features, len(x_train.columns))
    best_idx = val_score.index(np.max(val_score))
    return features[best_idx]

  def _plot(self, train_score, val_score, features, len_col):
    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(len_col)+1, train_score, label="Train")
    plt.plot(np.arange(len_col)+1, val_score, label="Val")
    plt.ylabel("Accuracy" if self.model_type == "classifier" else "reverse loss")
    plt.xlabel("number of faatures")
    plt.xticks(np.arange(1, 1+len_col))
    plt.legend(fontsize=12)
    plt.grid()
    type_ = "F value" if self.model_type == "classifier" else "F regression"
    plt.title(f"features selection by using {type_}")


class Wrapper:

  def __doc__(self):
     '''
     特長量を一つずつ足し合わせることで最も検証結果の良かった特長量の組み合わせを取り出す。
     パッケージ: mlxtend (pip install mlxtend)
     '''

  def __init__(self):
    self.model = None 
    self.x_train = []
    self.y_train = []

  def fit(self, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series], 
          forward: bool=True, floating: bool=False, verbose: int=2, scoring="accuracy", cv: int=5, n_jobs=-1) -> List[str]:
    self.model = sfs(LogisticRegression(), k_features=(1, len(x_train.columns)), forward=forward, 
                     floating=floating, verbose=verbose, scoring=scoring, cv=cv, n_jobs=n_jobs).fit(x_train, y_train)
    self._plot()
    self.x_train = x_train 
    self.y_train = y_train 

    return list(self.model.k_feature_names_)

  def _plot(self):
    fig = plot_sfs(self.model.get_metric_dict(), kind="std_dev")
    plt.title("Sequence forward generation (w, stdDev)")
    plt.grid()

  def predict(self, x_val: pd.DataFrame, y_val: Union[pd.DataFrame, pd.Series]):
    reg = LogisticRegression().fit(self.x_train, self.y_train)
    feature_sel = list(self.model.k_feature_names_)
    reg_ = LogisticRegression().fit(self.x_train[feature_sel], self.y_train)
    print(f"before score: {reg.score(x_val, y_val)}")
    print(f"after score: {reg_.score(x_val[feature_sel], y_val)}")
    

class Embedded:

  def __doc__(self):
    '''
    ラッソ回帰または決定木による重要性の高い特長量を抽出する。
    ラッソ回帰を用いる場合には、あらかじめ標準化が必要。
    '''

  def __init__(self):
    self.model = None 
    self.x_train = [] 
    self.y_train = [] 

  def fit(self, x_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame], 
          model_type: Union["ll", "tree"]="tree", C=1.0, solver: str="liblinear", threshold: str="mean",
          n_estimater: int=100, min_samples_leaf: int=40, random_state: int=0) -> List[str]:
    self.x_train, self.y_train = x_train, y_train 
    clf = LogisticRegression(C=C, panelty="ll", solver=solver) if model_type == "ll" else RandomForestClassifier(n_estimators=n_estimater, min_samples_leaf=min_samples_leaf, random_state=random_state)
    self.model = SelectFromModel(clf, threshold=threshold).fit(x_train, y_train)
    return x_train.columns[list(self.model.get_support())]

  def predict(self, x_val: pd.DataFrame, y_val: Union[pd.DataFrame, pd.Series]):
    reg = LogisticRegression().fit(self.x_train, self.y_train)
    feature_sel = self.x_train.columns[list(self.model.get_support())]
    reg_ = LogisticRegression().fit(self.x_train[feature_sel], self.y_train)
    print(f"before score: {reg.score(x_val, y_val)}")
    print(f"after score: {reg_.score(x_val[feature_sel], y_val)}")

