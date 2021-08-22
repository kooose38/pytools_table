import pandas as pd 
from sklearn.linear_model import LinearRegression
from typing import Union, Dict, Any
import pickle 
import os 
from sklearn.impute import KNNImputer


class RegressionForFillna:
  def __doc__(self):
    '''
    Predict missing values by regression analysis. 
    Make sure that the input data contains no missing values other than the columns you expect.
    Also, since it is a regression analysis, the data type is numerical.
    '''

  def __init__(self):
    self.model = None 
    self.column = ""

  def fit(self, train: pd.DataFrame, test: pd.DataFrame, column: Union[str, int]):
    '''
    欠損補完するカラム以外には欠損値を含んでいないこと。
    訓練データとテストデータを結合してから回帰分析を行うので、カラムサイズをそろえておくこと。
    '''
    self.column = column 
    train_c, test_c = train.copy(), test.copy()
    dataset = pd.concat([train_c, test_c], axis=0).reset_index().drop(["index"], axis=1)
    ix = dataset[column].isnull()

    x1 = dataset[~ix].drop(column, axis=1)
    y1 = dataset.loc[x1.index, column]
    self.model = LinearRegression().fit(x1, y1)

  def predict(self, data: pd.DataFrame) -> pd.DataFrame:
    '''fitした際と同じカラムを使うこと。'''
    ix = data[self.column].isnull()
    data[self.column+"_reg"] = pd.Series(self.model.predict(data[ix].drop(self.column, axis=1)), 
                                data[ix].index)
    data[self.column+"_reg"] = data[self.column+"_reg"].fillna(data[self.column])
    data[self.column] = data[self.column+"_reg"]
    data.drop(self.column+"_reg", axis=1, inplace=True)
    return data 

  def save(self, filepath: str):
    os.makedirs(filepath, exist_ok=True)
    model_path = os.path.join(filepath+"/"+f"reg_fillna_{self.column}.pkl")
    pickle.dump(self.model, open(model_path, "wb"))


class KNNForFillna(KNNImputer):
  def __doc__(self):
    '''
    Missing completion is performed from the neighbor relationship between the data by knn.
    '''

  def __init__(self):
    self.model = None 
    self.fill_col = ""
    self.columns = ""

  def fit(self, x_train: pd.DataFrame, x_test: pd.DataFrame, fill_col: Union[str, int], 
          n_neighbors: int=2):
    self.fill_col = fill_col
    train_c, test_c = x_train.copy(), x_test.copy()
    dataset = pd.concat([train_c, test_c], axis=0).reset_index().drop(["index"], axis=1)
    self.model = KNNImputer(n_neighbors=n_neighbors).fit(dataset)
    self.columns = x_train.columns   

  def predict(self, df: pd.DataFrame) -> pd.DataFrame:
    assert self.columns.tolist() == df.columns.tolist() 

    df[self.fill_col] = pd.DataFrame(self.model.transform(df), 
                                     index=df.index, 
                                     columns=df.columns)[self.fill_col]
    return df 
  
  def save(self, filepath: str):
    os.makedirs(filepath, exist_ok=True)
    model_path = os.path.join(filepath+"/"+f"knn_fillna_{str(self.fill_col)}.pkl")
    pickle.dump(self.model, open(model_path, "wb"))
    

