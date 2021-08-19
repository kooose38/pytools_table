import numpy as np 
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score

class CrossValScore:
    def __init__(self, x_train, y_train): # 学習データから検証データの作成する手法と評価平均の算出。
        self.x_train = x_train 
        self.y_train = y_train 
        self.a = np.random.randint(0, 100, 1)[0] # 分割のシード値
        
    def _evaluate(self, model, tr_x, tr_y, va_x, va_y):
        model.fit(tr_x, tr_y)
        predict = model.predict(va_x)
        if len(self.y_train.iloc[:, 0].value_counts().index) <= 20:
            loss = accuracy_score(predict, va_y.values.ravel())
        else:
            loss = mean_squared_error(va_y.values.ravel(), predict)
        return loss 
    
    def cross_val_score(self, model, cv=4):
        """
        通常の交差検証による検証結果の平均化
        """
        losses = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.a)
        for tr_idx, va_idx in kf.split(self.x_train):
            tr_x, va_x = self.x_train.iloc[tr_idx], self.x_train.iloc[va_idx]
            tr_y, va_y = self.y_train.iloc[tr_idx], self.y_train.iloc[va_idx]
            
            loss = self._evaluate(model, tr_x, tr_y, va_x, va_y)
            losses.append(loss)
        print(losses)
        print(np.mean(losses))
    
    def stratified_fold_cross_val_score(self, model, cv=4):
        """
        クラス分類タスクの時に、fold毎に含まれるクラスの割合を等しくする層化抽出で交差検証を行う。
        テストデータに含まれる各クラスの割合は、学習データに含まれるクラスの割合と同じという仮定に基づく。
        """
        losses = []
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.a)
        for tr_idx, va_idx in kf.split(self.x_train, self.y_train):
            tr_x, va_x = self.x_train.iloc[tr_idx], self.x_train.iloc[va_idx]
            tr_y, va_y = self.y_train.iloc[tr_idx], self.y_train.iloc[va_idx]
            
            loss = self._evaluate(model, tr_x, tr_y, va_x, va_y)
            losses.append(loss)
        print(losses)
        print(np.mean(losses))
        
    def group_fold_cross_val_score(self, model, target_col: str, cv=4):
        """
        テストデータにのみ存在するある値と同じように検証データを作成する。
        つまりカラムから、ある値を含むデータと含まないデータを分割し、未知の検証データからの交差検証を行う。
        """
        # カラムのデータからユニーク値を取得する
        target_col_id = self.x_train[target_col]
        unique_id = target_col_id.unique()
        
        losses = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.a)
        for tr_g, va_g in kf.split(unique_id):
            # train validation 分割
            tr_gs, va_gs = unique_id[tr_g], unique_id[va_g]
            # traget_colの分割によるデータの取得
            is_tr, is_va = target_col_id.isin(tr_gs), target_col_id.isin(va_gs)
            tr_x, va_x = self.x_train[is_tr], self.x_train[is_va]
            tr_y, va_y = self.y_train[is_tr], self.y_train[is_va]
            
            loss = self._evaluate(model, tr_x, tr_y, va_x, va_y)
            losses.append(loss)
        print(losses)
        print(np.mean(losses))
        
    def leave_one_out_cross_val_score(self, model):
        """
        cross_validationにおける`n_split`をデータの数の分だけ分割した評価を行います。
        ただし、検証データが１つのみなので検証結果の理解に注意する。
        そのため、学習データが極めて少ない場合など使える範囲は限定的。
        """
        loo = LeaveOneOut()
        losses = []
        for tr_idx, va_idx in loo.split(self.x_train):
            tr_x, va_x = self.x_train.iloc[tr_idx], self.x_train.iloc[va_idx]
            tr_y, va_y = self.y_train.iloc[tr_idx], self.y_train.iloc[va_idx]
            
            loss = self._evaluate(model, tr_x, tr_y, va_x, va_y)
            losses.append(loss)
        print(losses)
        print(np.mean(losses))