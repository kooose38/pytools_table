from typing import Any, Dict, Union, Tuple 
import eli5
from eli5.sklearn import PermutationImportance
import numpy as np

class VizPermitaionImportance:

  def __doc__(self):
    """
    package: eli5 (pip install eli5)
    support: sckit-learn xgboost lightboost catboost lighting 
    purpose: Visualize the importance of each feature quantity by giving a big picture explanation using a model
    """

  def __init__(self):
    pass 

  def show_weights(self, model: Any, importance_type: str="gain"):
    return eli5.show_weights(model, importance_type=importance_type)

  def show_feature_importance(self, model: Any, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series, np.ndarray], 
                              x_val: pd.DataFrame, y_val: Union[pd.DataFrame, pd.Series]) -> Tuple[object, object]:
    '''
    warnings: support sklearn model only!
    By trying both the training data and the validation data, you can lyric whether there is a difference in the distribution of the data.
    x_train, y_train -> x_val, y_val 
    '''
    perm_train = PermutationImportance(model).fit(x_train, y_train)
    fig1 =  eli5.show_weights(perm_train, feature_names=x_train.columns.tolist())
    perm_val = PermutationImportance(model).fit(x_val, y_val)
    fig2 = eli5.show_weights(perm_val, feature_names=x_val.columns.tolist())
    return fig1, fig2
