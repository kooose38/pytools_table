# !pip install -q xgboost 
# !pip install optuna
import xgboost as xgb 
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pandas as pd 
import numpy as np 
import pickle
import matplotlib.pyplot as plt 
import optuna 

class XGBoost:
    def __init__(self):
        #主要なメソッド
        # train() validationの作成とモデル評価、モデルの保存を行います
        # load_model() 保存したモデルを読み込み、返します
        # optuna() ベイズ最適化によるハイパーパラメータの探索
        pass
        
        
    def plot_feature(self, model):
        fig, ax = plt.subplots(figsize=(12, 12))
        xgb.plot_importance(model, max_num_features=10, height=0.8, ax=ax)
    def model(self,
              dtrain,
              dtest,
              y_train,
              x_test,
              num_round, 
              early_stopping_rounds, 
              x_val,
              y_val,
              param):
        if len(param) == 0:
            #デフォルトの設定
            param = {
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
        
        #二値分類によるパラメータ指定
        if len(y_train.value_counts().index) <= 2:
            param["objective"] = "binary:logistic"
            param["eval_metric"] = "logloss"
        #多値分類   
        elif len(y_train.value_counts().index) <= 10:
            param["objective"] = "multi:softprob"
            param["eval_metric"] = "mlogloss"
            param["num_class"] = len(y_train.value_counts().index)
        #回帰        
        else:
            param["objective"] = "reg:squarederror"
            
  
     
        watch_list = [(dtrain, "train"), (dtest, "eval")]
        model = xgb.train(param,
                          dtrain,
                          num_round,
                          evals=watch_list, 
                          early_stopping_rounds=early_stopping_rounds)
        #テストデータへの予測とファイルの出力を行います
        #分類、回帰への変換は自動で行います
        if x_test.shape[0] > 0:
            pred_probe = model.predict(xgb.DMatrix(x_test), ntree_limit=model.best_ntree_limit)
            if len(y_train.value_counts().index) <= 2:
                pred = np.array(pred_probe)
                pred = np.where(pred >= 0.5, 1, 0)
            elif len(y_train.value_counts().index) <= 10:
                pred = pred_probe.argmax(axis=1)
            else:
                pred = np.array(pred_probe)
                
            x_test["predict"] = pred 
            print("テストの予測値をファイルに出力します")
            x_test.to_csv("./submission/y_test_submission_xgb.csv", index=False)
            
        pred_probe = model.predict(xgb.DMatrix(x_val), ntree_limit=model.best_ntree_limit)
        #検証データに対して評価をします
        if len(y_train.value_counts().index) <= 2:
            pred = np.array(pred_probe)
            pred = np.where(pred >= 0.5, 1, 0)
            print(f"正解率: {accuracy_score(y_val, pred)}%")

        elif len(y_train.value_counts().index) <= 10:
            pred = pred_probe.argmax(axis=1)
            pred = np.array(pred)
            print(f"正解率: {accuracy_score(y_val, pred)}%")

        else:
            pred = np.array(pred_probe)   
            print(f"Loss: {mean_squared_error(y_val, pred)}")
            
        self.plot_feature(model)  #特長量の重要度の可視化を行います  
        self.save_model(model) #モデルの保存
        
    def save_model(self, model):
        print("モデルの保存を行います。")
        filename = "./model/xgboost.sav"
        pickle.dump(model, open(filename, 'wb'))

    def train(self,  
              x_train,
              y_train,
              x_val=pd.DataFrame(), #検証データが決まっている場合の指定
              y_val=pd.DataFrame(), 
              x_test=pd.DataFrame(), 
              param={}, 
              early_stopping_rounds=50,
              num_round=1000):
        
        if x_val.shape[0] > 0:
            dtrain = xgb.DMatrix(x_train, y_train)
            dtest = xgb.DMatrix(x_val, y_val)

        else:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=0)
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_val, label=y_val)
        self.model(dtrain, dtest, y_train, x_test, num_round, early_stopping_rounds, x_val, y_val, param)
            
    def load_model(self, filename):
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model
        
    
    def optuna(self, x_train, y_train, x_val=pd.DataFrame(), y_val=pd.DataFrame(), early_stopping_rounds=50):
        #objective関数に引数を渡すためコールバック関数でラップしてます
        def objective_variable(dtrain, dtest, y_val, early_stopping_rounds):

            def objective(trial):

                max_depth = trial.suggest_int("max_depth", 3, 9)
                colsample_bytree = trial.suggest_loguniform("colsample_bytree", .1, 1.0)
                min_child_weight = trial.suggest_loguniform("min_sample_weight", 1, 10)
                gamma = trial.suggest_loguniform("gamma", 1e-8, 1.0)
                subsample = trial.suggest_loguniform("subsample", .6, 1.0)
                alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
                eta = trial.suggest_loguniform("eta", 0.001, 0.1)
                n_lambda = trial.suggest_loguniform("lambda", 1e-6, 10.0)

                param = {
                    "eta": eta,
                    "silent": 1,
                    "random_state": 0,
                    "gamma": gamma, 
                    "alpha": alpha, 
                    "max_depth": max_depth,
                    "lambda": n_lambda,
                    "min_child_weight": min_child_weight,
                    "colsample_bytree": colsample_bytree,
                    "subsample": subsample
                }

                #二値分類によるパラメータ指定
                if len(y_val.value_counts().index) <= 2:
                    param["objective"] = "binary:logistic"
                    param["eval_metric"] = "logloss"

                elif len(y_val.value_counts().index) <= 10:
                        param["objective"] = "multi:softprob"
                        param["eval_metric"] = "mlogloss"
                        param["num_class"] = len(y_val.value_counts().index)

                else:
                    param["objective"] = "reg:squarederror"
                    print(param)
                num_round = 100
                watch_list = [(dtrain, "train"), (dtest, "eval")]
                model = xgb.train(param,
                                  dtrain,
                                  num_round,
                                  evals=watch_list,
                                  early_stopping_rounds=early_stopping_rounds)


                result = model.predict(xgb.DMatrix(x_val), ntree_limit=model.best_ntree_limit)
                if len(y_val.value_counts().index) <= 2:
                    pred = np.array(result)
                    pred = np.where(pred >= 0.5, 1, 0)
                    result = log_loss(y_val, pred)

                elif len(y_val.value_counts().index) <= 10:
                    #pred = result.argmax(axis=1)
                    result = log_loss(y_val, np.array(result))
                else:
                    result = mean_squared_error(y_val, np.array(result))

                return result #lossの最小化を目標値とします
            return objective

        # 検証データが指定のものを使う場合
        if x_val.shape[0] > 0:
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_val, label=y_val)
        else:
            random_state = np.random.randint(1, 59, 1)[0] #データの偏りを防ぐため、seed値のランダム化
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=random_state)
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_val, label=y_val)
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective_variable(dtrain, dtest, y_val, early_stopping_rounds), n_trials=50)
        
        print(f"ロス値: {study.best_value}")
        print(f"パラメータ: {study.best_params}")
        print(f"trial: {study.best_trial}")
        
        for i in range(4):
            if i == 0:
                #最適化の履歴を確認するplot_optimization_historyメソッドです。
                #縦軸が目的変数、横軸が最適化のトライアル数になってます。オレンジの折れ線が最良の目的変数の値となっており、何回目のトライアルでベストパラメータが出たのかわかりやすくなってます。
                fig = optuna.visualization.plot_optimization_history(study)
            if i == 1:
                #各パラメータの値と目的変数の結果をプロットするメソッドです
                fig = optuna.visualization.plot_slice(study)
            if i == 2:
                #各パラメータにおける目的変数の値がヒートマップで表示されます。
                fig = optuna.visualization.plot_contour(study)
            if i == 3:
                #どのパラメータが効いていたか表すメソッドです。
                fig = optuna.visualization.plot_param_importances(study)
            fig.show()
    
            