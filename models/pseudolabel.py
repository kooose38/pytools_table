from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt 
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
import pandas as pd 

class PseudoLabeling:
    def __init__(self, n_splits=4, params={}, num_round=100):
        """
        テストデータの予測値を使って再度学習を行い、kfoldによる残りのテストデータを最終予測値とします。
        訓練データがテストデータよりも少ない場合や、テストデータの情報を使いたいに有効です。
        時系列データに対しては未来の情報を使ってしまうのでリークが起きます。
        
        ただし以下の欠点があります
        1. 確かでない予測値に従って学習し、このモデルから最終予測とするので過学習しやすい
        2. kfoldによるモデル作成を別個行っているのでパラメータチューニングが行いにくい
        """
        self.n_splits = n_splits # kfoldの分割回数 4-10がいいとされる
        self.param = params # xgboost parameterの指定、目的関数は自動で設定します
        self.num_round = num_round # 学習回数
        if len(self.param) == 0: 
            #デフォルトの設定
            self.param = {
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
        print(f"parameters: \n{self.param}")
        
    def _plot_feature(self, model, idx):
        fig, ax = plt.subplots(figsize=(8, 8))
        xgb.plot_importance(model, 
                            max_num_features=10,
                            height=0.8,
                            ax=ax,
                            title=f"xgb-{idx+1} feature_importance top 10")
        
    def train(self, train_x, train_y, x_test):
        
        x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, random_state=0)
        
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        
        #二値分類によるパラメータ指定
        if len(y_train.value_counts().index) <= 2:
            self.param["objective"] = "binary:logistic"
            self.param["eval_metric"] = "logloss"
        #多値分類   
        elif len(y_train.value_counts().index) <= 10:
            self.param["objective"] = "multi:softprob"
            self.param["eval_metric"] = "mlogloss"
            self.param["num_class"] = len(y_train.value_counts().index)
        #回帰        
        else:
            self.param["objective"] = "reg:squarederror"
            
        watch_list = [(dtrain, "train"), (dval, "eval")]
        model = xgb.train(self.param,
                         dtrain,
                         self.num_round,
                         evals=watch_list,
                         early_stopping_rounds=20)
        
        preds, idx = [], []
        kf = KFold(n_splits=self.n_splits, random_state=0, shuffle=True)
        for i, (va_idx, te_idx) in enumerate(kf.split(x_test)):
            kf_val, kf_test = x_test.iloc[va_idx], x_test.iloc[te_idx]
            # 全体のうち3/4のテストデータの予測を行い、次の学習時の目的変数とします
            pred = model.predict(xgb.DMatrix(kf_val), ntree_limit=model.best_ntree_limit)
            if len(y_train.value_counts().index) <= 2:
                pred = np.array(pred)
            elif len(y_train.value_counts().index) <= 10:
                pred = np.array(pred)
                pred = pred.argmax(axis=1)
            else:
                pred = np.array(pred)
            # テストデータの一部予測値を目的変数として訓練モデルとする
            pred = pd.DataFrame({train_y.columns[0]: pred})
            # テストデータ3/4と、訓練データの組み合わせ
            x_concat_train_kftest = pd.concat([train_x, kf_val])
            y_concat_train_kftest = pd.concat([train_y, pred])
            x2_train, x2_val, y2_train, y2_val = train_test_split(x_concat_train_kftest, y_concat_train_kftest, random_state=0)
            
            
            d2train = xgb.DMatrix(x2_train, label=y2_train)
            d2val = xgb.DMatrix(x2_val, label=y2_val)
            
            if i == 0:
                print("-"*100)
            
            watch_list = [(d2train, "train"), (d2val, "eval")]
            print(f"{i+1}/{self.n_splits} pseudo labelによる学習")
            # 残りのテストデータ1/4を予測するためのモデル
            model2 = xgb.train(self.param,
                             d2train,
                             self.num_round, 
                             evals=watch_list,
                             early_stopping_rounds=20)
            # 決定木による特長量の重要度を可視化
            self._plot_feature(model2, i)

            pred = model2.predict(xgb.DMatrix(kf_test), ntree_limit=model2.best_ntree_limit)
            val_predict = model2.predict(xgb.DMatrix(x2_val), ntree_limit=model2.best_ntree_limit)
            # 予測を行いつつ、検証へのロス算出をする
            if len(y_train.value_counts().index) <= 2:
                pred = np.array(pred)
                pred = np.where(pred >= .5, 1, 0)
                
                predict_val = np.array(val_predict)
                predict_val = np.where(predict_val >= .5, 1, 0)
                
                loss = log_loss(np.where(y2_val.values.ravel() >= .5, 1, 0), 
                                predict_val)
                acc = accuracy_score(np.where(y2_val.values.ravel() >= .5, 1, 0), predict_val)
                print(f"{i+1}/{self.n_splits} 正解率: {acc*100}%")
                print(f"{i+1}/{self.n_splits} Loss: {loss}")
                
            elif len(y_train.value_counts().index) <= 10:
                pred = np.array(pred)
                pred = pred.argmax(axis=1)
                
                predict_val = np.array(val_predict)
                predict_val = predict_val.argmax(axis=1)
                loss = log_loss(y2_val.values.ravel(), predict_val)
                acc = accuracy_score(y2_val.values.ravel(), predict_val)
                print(f"{i+1}/{self.n_splits} 正解率: {acc*100}%")
                print(f"{i+1}/{self.n_splits} Loss: {loss}")
                
            else:
                pred = np.array(pred)
                predict_val = np.array(val_predict)
                
                loss = mean_squared_error(y2_val.values.ravel(), predict_val)
                print(f"{i+1}/{self.n_splits} Loss: {loss}")
            print("-"*100)

          
            preds.append(pred)
            idx.append(te_idx)
        # fold分割のindex整理
        te_idxs = np.concatenate(idx)
        preds_test = np.concatenate(preds, axis=0)
        order = np.argsort(te_idxs)
        predict_test = preds_test[order]
        
        x_test["predict_pseudolabel"] = predict_test 
        x_test.to_csv("./submission/x_test_submission_pseudolabel.csv", index=False)
            
        