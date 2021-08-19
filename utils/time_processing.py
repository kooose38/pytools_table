import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
class TimeSeriesSplit_:
    def __init__(self):
        pass
    def create_validation(self, x_train, y_train, date_col=[], periods=5, index=False):
        """
        時系列に従い、訓練と検証の分割をします。
        ここではテストデータへの影響が最近の情報に近似していることを仮定できるデータに対して有効的です。
        """
        if index:
            date_col.append(x_train.index.name)
            x_train = x_train.reset_index()
            
        x_train["periods"] = pd.cut(x_train[date_col[0]], periods, labels=False)
   
        tr = x_train["periods"] < periods-1
        va = x_train["periods"] == periods-1
        x_train_copy, y_train_copy = x_train[tr], y_train[tr]
        x_val, y_val = x_train[va], y_train[va] 
        x_train_copy.drop(["periods"], axis=1, inplace=True)
        x_val.drop(["periods"], axis=1, inplace=True)
        if index:
            x_train_copy.set_index(date_col, inplace=True)
            x_val.set_index(date_col, inplace=True)
            x_train_copy.sort_index(ascending=True)
            x_val.sort_index(ascending=True)
        return x_train_copy, x_val, y_train_copy, y_val
    
    def cross_val_score(self, model, x_train, y_train, date_col=[], periods=5):
        """
        時系列に従って訓練、検証分割後に交差検証によってモデルのロスを算出します。
        訓練データには検証以降の未来のデータを含めずに学習が進むので、主に時系列性が強い回帰問題に用います。
        
        途中で`_fit_transform`で時系列処理を施していますが、データに応じてコードを変更してください。
        """
        x_train_c = x_train.copy()
        y_train_c = y_train.copy()
        x_train_c["periods"] = pd.cut(x_train_c[date_col[0]], periods, labels=False)
        x_train_c = self._fit_transform(x_train_c, date_col[0])
        
        periods_list = np.arange(1, periods).tolist()
        scores = []
        for p in periods_list:
            tr = x_train_c["periods"] < p 
            va = x_train_c["periods"] == p 
            x_tr, y_tr = x_train_c[tr], y_train_c[tr]
            x_val, y_val = x_train_c[va], y_train_c[va]
            x_tr = x_tr.iloc[:, :-1] # except periods column
            x_val = x_val.iloc[:, :-1]
            
            model.fit(x_tr, y_tr)
            pred = model.predict(x_val)
            val_score = mean_squared_error(pred, y_val.values.ravel()) # 指標に応じて損失関数の変更
            scores.append(val_score)
        print(scores)
        print(f"total average validation Loss: {np.array(scores).mean():.3f}")
            
    def _fit_transform(self, x_train, date_col):
        """
        学習が進むような最低限のデータ変換
        """
        x_train["year"] = x_train[date_col].dt.year 
        x_train["month"] = x_train[date_col].dt.month
        x_train["days"] = x_train[date_col].dt.day
        x_train.drop([date_col], axis=1, inplace=True)
        return x_train 
    
