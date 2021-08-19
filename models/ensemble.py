from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 

#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

#Models
import plotly.express as px 
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
from sklearn.metrics import log_loss, accuracy_score

class Stacking:
    def __init__(self):
        """
        主に分類問題で使用します。
        時系列データの依存が強い、テストデータの分布が異なる場合は有効ではありません。
        複数の学習器をk分割して訓練テストデータの予測。
        1. 訓練にはリークしないように目的変数を使いません
        2. テストデータには予測の平均をとります
        
        二層目で最終的な評価を行います。
        
        なお、以下の二点の.csvに書き出します
        1. それぞれの学習規の予測を組み合わせた一層目の出力
        2. 1のデータによって学習したモデル毎のテストデータに対する最終予測値
        """
        pass
    
    def predict_cv(self, model, x_train, y_train, x_test, names, flag=False):
        preds = []
        preds_test = []
        idx = []
        kf = KFold(n_splits=4, shuffle=True, random_state=0)
        
        for i, (tr_idx, va_idx) in enumerate(kf.split(x_train)):
            x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model.fit(x_tr, y_tr)
            pred = model.predict(x_va)
            preds.append(pred)
            pred_test = model.predict(x_test)
            preds_test.append(pred_test)
            idx.append(va_idx)

        va_idxs = np.concatenate(idx)
        preds = np.concatenate(preds, axis=0)
        order = np.argsort(va_idxs)
        preds_train = preds[order]

        preds_test = np.mean(preds_test, axis=0)
        
        #二層目
        if flag:
            print(f"アンサンブル学習のデータから{names}による予測を出力します。")
            print(f"Model: {names} -- Loss: {log_loss(y_train, preds_train)}")
            print(f"Model: {names} -- accuracy: {accuracy_score(y_train, preds_train)}")
            x_test_copy = x_test.copy()
            x_test_copy["predict"] = preds_test
            x_test_copy.to_csv(f"./submission/x_test_ensemble_{names}.csv", index=False)
            return names, log_loss(y_train, preds_train), accuracy_score(y_train, preds_train)
        #一層目
        else:
            return preds_train, preds_test
    def create_model(self):
        clfs = []
        seed = 3

        clfs.append(("XGBClassifier",
                     Pipeline([
                               ("XGB", XGBClassifier(n_jobs=-1, random_state=42))]))) 
        clfs.append(("SVC", 
                    Pipeline([
                              ("SVC", SVC(random_state=42))]))) 
#         clfs.append(("LogisticRegression", 
#                     Pipeline([
#                               ("LogisticRegression", LogisticRegression(random_state=42))]))) 

        clfs.append(("SGD", 
                    Pipeline([
                              ("SGD", SGDClassifier(random_state=42))]))) 
        clfs.append(("LinearSVC", 
                    Pipeline([
                              ("LinearSVC", LinearSVC(random_state=42))]))) 

        clfs.append(("DecisionTreeClassifier", 
                     Pipeline([
                               ("DecisionTrees", DecisionTreeClassifier(random_state=42))]))) 

        clfs.append(("RandomForestClassifier", 
                     Pipeline([
                               ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1, 
                                                                       random_state=42))]))) 

        clfs.append(("GradientBoostingClassifier", 
                     Pipeline([
                               ("GradientBoosting", GradientBoostingClassifier(n_estimators=200,
                                                                               random_state=42))]))) 

        clfs.append(("RidgeClassifier", 
                     Pipeline([
                               ("RidgeClassifier", RidgeClassifier(random_state=42))])))

        clfs.append(("BaggingRidgeClassifier",
                     Pipeline([
                               ("BaggingClassifier", BaggingClassifier(n_jobs=-1, random_state=42))])))

        clfs.append(("ExtraTreesClassifier",
                     Pipeline([
                               ("ExtraTrees", ExtraTreesClassifier(n_jobs=-1, random_state=42))])))
        return clfs
    
    def train(self, x_train, y_train, x_test):
        '''
        return:
        df_train: 一層目のモデル毎の予測値
        df_test: 一層目のテストデータに対する予測値の平均
        '''
        
        clfs = self.create_model()
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        
        for names, model in clfs:
            train, test = self.predict_cv(model, x_train, y_train, x_test, names)
            df_train[f"{names}_predict"] = train 
            df_test[f"{names}_predict"] = test 
        
        result_name, result_score, result_loss = [], [], []
        for names, model in clfs:
            name, loss, accuracy = self.predict_cv(model, 
                            df_train,
                            y_train,
                            df_test,
                            names,
                            flag=True)
            result_name.append(names)
            result_score.append(accuracy)
            result_loss.append(loss)
        try:    
            for i in range(2):
                if i == 0:
                    fig = px.line(x=result_name,
                                 y=result_loss,
                                 title="result for enusemble Loss values")
                    fig.show()
                if i == 1:
                    fig = px.bar(x=result_name,
                                 y=result_score, 
                                 title="result for enusemble accuracy scores")
                    fig.show()
        except:
            print("not modules plotly")
        finally:
            print("アンサンブル学習による予測をまとめたファイルの出力を行います。")
            return df_train, df_test
#         df_train.to_csv("./transform/x_train_ensemble.csv", index=False)
#         df_test.to_csv("./transform/x_test_ensemble.csv", index=False)
        
        
                
        
        
        