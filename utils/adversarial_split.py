import pandas as pd 
from sklearn.linear_model import LogisticRegression
import numpy as np 

class AdversarialValidation:
    def __init__(self, threshold=.5):
        # テストデータ分布を表現する際の閾値
        self.threshold = threshold 
    def fit_transform(self, x_train, y_train, x_test):
        """
        学習データとテストデータを結合して、テストデータか否かを目的変数とする二値分類を行うことで、
        学習データとテストデータの分布が同じかどうかを判断とする手法。
        
        学習データとテストデータの分布が大きく異なる、その差異の要因が不明時の分析としても有効的である。
        入力データは与えられたデータをそのまま用いる方がよい。
        """
        # trainデータの長さを保持する
        n_train = x_train.shape[0]
        # 元データの上書きを防ぐためデータ複製
        x_train_ = x_train.copy()
        x_test_ = x_test.copy()
        y_train_ = y_train.copy()
        # テストデータらしさを表現するための疑似的な正解ラベル作成
        x_train_["val"] = 0 
        x_test_["val"] = 1 
        # ラベルが異なると学習ができないのでここでテストを挟む
        assert len(x_train.columns) == len(x_test.columns)
        # 疑似の訓練データセット
        dataset = pd.concat([x_train_, x_test_]).reset_index().drop(["index"], axis=1)
        x_, t_ = dataset.drop(["val"], axis=1), dataset[["val"]]
        log = LogisticRegression().fit(x_, t_)
        # 閾値に従って0/1に分割する
        # 検証データを多く得たい時は閾値を小さく設定すること
        y_ = log.predict_proba(x_)[:, 1].ravel()
        y_ = np.array(y_)
        y_ = np.where(y_ >= self.threshold, 1, 0)
        # 学習データのうち、テストデータと予測されたもの(テストデータの分布に近い)を検証データとする
        dataset["y"] = y_
        # テストデータの排除
        train = dataset[:n_train].drop(["val"], axis=1)
        y_train_["y"] = train["y"]
        # 不要なカラムの削除と検証の作成
        x_train__, x_val = train[train["y"] == 0].reset_index().drop(["y", "index"], axis=1), train[train["y"] == 1].reset_index().drop(["y", "index"], axis=1)
        y_train__, y_val = y_train_[y_train_["y"] == 0].reset_index().drop(["y", "index"], axis=1), y_train_[y_train_["y"] == 1].reset_index().drop(["y", "index"], axis=1)
        return x_train__, x_val, y_train__, y_val 
        