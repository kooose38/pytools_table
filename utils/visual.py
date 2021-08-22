import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt

class Viz:
  def __init__(self):
    pass 

  def pairplot_scatter(self, df_train: pd.DataFrame, hue: str=None, bins: int=20, alpha: float=0.2) -> object:
    metadata = {
        "data": df_train,
        "plot_kws": {"alpha": alpha},
    }
    if hue is not None:
      metadata["hue"] = hue 
    else:
      metadata["diag_kws"] = {"bins": bins}
    return sns.pairplot(**metadata)

  def pairplot_reg(self, df_train: pd.DataFrame, hue: str=None) -> object:
    metadata = {
        "data": df_train,
        "kind": "reg"
    }
    if hue is not None:
      metadata["hue"] = hue 
    return sns.pairplot(**metadata)

  def pie(self, df_train: pd.DataFrame) -> object:
    category = df_train.select_dtypes(["category", "bool", "object"]).columns.to_list()
    fig = df_train[[category]].plot.pie(subplots=True, figsize=(11, 6))
    return fig 

  def box(self, df_train: pd.DataFrame) -> object:
    plt.figure(figsize=(14, 6))
    df_train.boxplot()
    plt.xticks(rotation=70)
    plt.show()
