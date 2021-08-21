from typing import Dict, Any, Union 
import pandas as pd 
import xgboost as xgb 
import os 
import uuid 
import logging 
import matplotlib.pyplot as plt
import pickle 
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
import optuna 
import numpy as np 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )

class XGBoost:
  def __init__(self, params: Dict[str, Any]={}):
    self.params = {
        "eta": 0.1, #学習率
        "silent": 1,
        "random_state": 0, #モデルに再現性を持たせるため固定
        "gamma": 0.0, #決定木分岐時に最低限減らすべき目的関数
        "alpha": 0.0,  #Lasso正則化
        "max_depth": 5, #決定木の深さ
        "lambda": 1.0, #Ridge正則化
        "min_child_weight": 1, #葉を分岐する軒最低限必要なデータ数
        "colsample_bytree": 0.8, #特長量の列の割合
        "subsample": 0.8 #特長量の行の割合
    }

    self.model = None 
    self.methods = ""
    self.dtrain = None 
    self.dval = None

    # for optuna validation data 
    self.x_val: Union[pd.DataFrame, None] = None 
    self.y_val: Union[pd.DataFrame, pd.Series, None] = None  

    # learning optuna object
    self.study = None 

    if len(params):
      self._load_params(params)
  

  def _load_paramas(self, paramas: Dict[str, Any]) -> Dict[str, Any]:
    for name, value in params.items():
      if name in self.params:
        self.params[name] = value 
    return self.paramas

  def _create_data(self, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series], x_val: pd.DataFrame, y_val: Union[pd.DataFrame, pd.Series]):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    return dtrain, dval 

  def fit(self, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series], x_val: pd.DataFrame, y_val: Union[pd.DataFrame, pd.Series], 
          early_stopping_rounds: int=50, num_rounds: int=100, 
          methods: Union["classifier", "multi-classifier", "regression"]="classifier", tag_size: int=3):
    # setting params for methods name 
    self.methods = methods 
    if methods == "classifier":
      self.params["objective"] = "binary:logistic"
      self.params["eval_metric"] = "logloss"
    elif methods == "multi-classifier":
      self.params["objective"] = "multi:softprob"
      self.params["eval_metric"] = "mlogloss"
      self.params["num_class"] = tag_size 
    elif methods == "regression":
      self.params["objective"] = "reg:squarederror"

    self.x_val = x_val 
    self.y_val = y_val 

    dtrain, dval = self._create_data(x_train, y_train, x_val, y_val)
    self.dtrain, self.dval = dtrain, dval 
    watch_list = [(dtrain, "train"), (dval, "eval")]

    self.model = xgb.train(self.params, dtrain, num_rounds, evals=watch_list, early_stopping_rounds=early_stopping_rounds)
    return self.model 

  def save(self, filepath: str="models"):
    id = str(uuid.uuid4())[:4]
    os.makedirs(filepath, exist_ok=True)
    model_path = os.path.join(filepath+"/"+f"xgb_{id}.pkl")
    pickle.dump(self.model, open(model_path, "wb"))
    logger.info(f"complete saving model file path : {model_path}")

  def show_feature_impotrance(self, max_num_features: int=10):
    fig, ax = plt.subplots(figsize=(10, 10))
    xgb.plot_importance(self.model, max_num_features=max_num_features, height=0.8, ax=ax)

  def parameter_chunning(self, early_stopping_rounds: int=40, num_rounds: int=100, seed: int=0, n_trials: int=30) -> Dict[str, Any]:
    """
    Use optuna to search for hyperparameters.
    The training data and validation data are stored in the constructor, so no arguments are required.
    Please refer to the metadata function for details of various parameters.
    """
    def objective_variable(early_stopping_rounds: int, num_rounds: int) -> float:

            def objective(trial) -> float:

                max_depth = trial.suggest_int("max_depth", 3, 9)
                colsample_bytree = trial.suggest_loguniform("colsample_bytree", .1, 1.0)
                min_child_weight = trial.suggest_loguniform("min_sample_weight", 1, 10)
                gamma = trial.suggest_loguniform("gamma", 1e-8, 1.0)
                subsample = trial.suggest_loguniform("subsample", .6, 1.0)
                alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
                eta = trial.suggest_loguniform("eta", 0.001, 0.1)
                n_lambda = trial.suggest_loguniform("lambda", 1e-6, 10.0)

                self.params["max_depth"] = max_depth
                self.params["colsample_bytree"] = colsample_bytree
                self.params["min_child_weight"] = min_child_weight
                self.params["gamma"] = gamma
                self.params["subsample"] = subsample
                self.params["alpha"] = alpha
                self.params["eta"] = eta
                self.params["n_lambda"] = n_lambda

                watch_list = [(self.dtrain, "train"), (self.dval, "eval")]
                model = xgb.train(self.params,
                                  self.dtrain,
                                  num_rounds,
                                  evals=watch_list,
                                  early_stopping_rounds=early_stopping_rounds)

                result = model.predict(xgb.DMatrix(self.x_val), ntree_limit=model.best_ntree_limit)
                if self.methods == "classifier":
                    result = log_loss(self.y_val, np.array(result))
                elif self.methods == "multi-classifier":
                    result = result.argmax(-1)
                    result = log_loss(self.y_val, np.array(result))
                else:
                    result = mean_squared_error(self.y_val, np.array(result))
                return result
            return objective

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective_variable(early_stopping_rounds, num_rounds), n_trials=n_trials)

    result = {
        "best_loss": study.best_value,
        "best_parameters": study.best_params,
        "best_trial": study.best_trial
    }

    self.study = study 
    return result 

  def show_optuna_viz(self):
    if self.study is not None:
      for i in range(4):
        if i == 0:
          fig = optuna.visualization.plot_optimization_history(self.study)
        if i == 1:
          fig = optuna.visualization.plot_slice(self.study)
        if i == 2:
          fig = optuna.visualization.plot_contour(self.study)
        if i == 3:
          fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()
    else:
      raise NotImplementedError

  def show_weights(self, importance_type: str="gain"):
    '''show Permutation Importance features'''
    import eli5
    eli5.show_weights(self.model, importance_type=importance_type)


    

    

