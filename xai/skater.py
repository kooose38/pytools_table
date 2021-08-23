from skater.core.explanations import Interpretation
from skater.model import InMemoryModel 
from skater.util.logger import _INFO
from sklearn.metrics import accuracy_score
from typing import Any, Union, List 
import pandas as pd 
import numpy as np 
from PIL import Image 

class Surrogate:

  def __doc__(self):
    '''
    package: skater (pip install git+https://github.com/oracle/Skater.git)
    support: sckit-leran model only 
    purpose: Make it descriptive by creating a simplified surrogate model. 
    By reducing the number of branches, especially in the decision tree, conditional branching that is easy to interpret can be obtained.
    '''

  def __init__(self):
    self.explainer = None 
    self.model_type = ""

  def fit(self, model: Any, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series, np.ndarray], 
                target_names: List[str], model_type: str="classifier",
                seed: int=0, max_depth: int=3, use_oracle: bool=True, prune: str="post", scorer_type: str="default"):
    self.model_type = model_type 
    interpreter = Interpretation(x_train, feature_names=x_train.columns)
    model_inst = InMemoryModel(model.predict, examples=x_train, model_type=model_type, 
                               unique_values=[0, 1], feature_names=x_train.columns, 
                               target_names=target_names, log_level=_INFO)
    self.explainer = interpreter.tree_surrogate(oracle=model_inst, seed=seed, max_depth=max_depth)
    self.explainer.fit(x_train, y_train, use_oracle=use_oracle, prune=prune, scorer_type=scorer_type)
    return self.explainer 

  def predict(self, x_train: pd.DataFrame, train_pred: np.ndarray) -> float:
    '''
    Compare the predictions of the original model with the predictions of the simplified decision tree model.
    It returns the reproducibility of the original model
    
    train_pred: [0, 0, 1, 0, 0, 1, ...]
    '''
    if self.model_type == "classifier":
      return accuracy_score(train_pred, self.explainer.predict(x_train))

  def plot(self, filename: str="sample_skater_tree.png"):
    self.explainer.plot_global_decisions(colors=["coral", "lightsteelblue", "darkhaki"], 
                                         file_name=filename)
    img = Image.open(filename)
    return img 
