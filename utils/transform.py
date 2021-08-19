###------------------------------------交差検証---------------------------------------------###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

#Models
import plotly.express as px 
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
from sklearn.metrics import log_loss

class ModelPipline:
    def __init__(self, 
                 x_train,
                 y_train,
                 x_val=pd.DataFrame(),
                 y_val=pd.DataFrame()):
        """
        データはPandas.DataFrame型であること
        """
        self.names = None 
        self.results = None 
        self.x_train = x_train 
        self.y_train = y_train 
        self.x_val = x_val 
        self.y_val = y_val 
        
        self.forward()
        
    def create_model(self):
        clfs = []
        seed = 3

        clfs.append(("LogReg", 
                     Pipeline([
                               ("LogReg", LogisticRegression(n_jobs=-1, random_state=42))])))

        clfs.append(("XGBClassifier",
                     Pipeline([
                               ("XGB", XGBClassifier(n_jobs=-1, random_state=42))]))) 
        clfs.append(("SVC", 
                    Pipeline([
                              ("KNN", SVC(random_state=42))]))) 
        clfs.append(("LogisticRegression", 
                    Pipeline([
                              ("LogisticRegression", LogisticRegression(random_state=42))]))) 

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
    def forward(self):
        """
        時系列や、分布の違いによりあらかじめ決められたvalidationデータで検証したい場合,validtionをTrueにします。
        それ以外ではtrainデータによる交差検証を行います。
        validation: boolean 
        x_val: 
        y_val:
        """
        if self.x_val.shape[0] > 0:
            if (self.x_val.shape[0] <= 0) or (self.y_val.shape[0] <= 0):
                print("x_val, y_valの指定をしてください")
            else:
                clfs = self.create_model()
                
                names, results = [], []
                
                for name, model in clfs:
                    model.fit(self.x_train, self.y_train)
                    score = model.score(self.x_val, self.y_val)
                    results.append(score)
                    names.append(name)
                    
                    print(f"model: {name} -- scoring: {score}")
                    
                self.names = names 
                self.results = results 
                
                label = []
                for re in results:
                    if re >= .9:
                        label.append("high")
                    elif re > .7:
                        label.append("medium")
                    else:
                        label.append("low")
                self.plotly(label)
        else:
        
            x_train = self.x_train 
            y_train = self.y_train 

            clfs = self.create_model()

            scoring = 'roc_auc'
            n_folds = 4

            results, names  = [], [] 

            # x_train, y_train はPandas.DataFrame型で渡す
            for name, model  in clfs:
                result = []
                # 交差検証
                kfold = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
                for tr_idx, va_idx in kfold.split(x_train):
                    x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
                    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                    model.fit(x_tr, y_tr)
                    score = model.score(x_va, y_va)
                    result.append(score)

                names.append(name)
                result = np.array(result)
                results.append(result.mean())    
                msg = "model: {} -- scoring : {:.3f}".format(name, result.mean())
                print(msg)

            self.names = names 
            self.results = results
            label = []
            for re in results:
                if re >= .9:
                    label.append("high")
                elif re > .7:
                    label.append("medium")
                else:
                    label.append("low")
            self.plotly(label)
            
    def plotly(self, label):
        """
        結果の可視化
        """
        if self.names != None:
            fig = px.bar(x=self.names,
                     y=self.results,
                     title="model scoring:", color=label)
            fig.show()  
            
###---------------------------------欠損処理クラス----------------------------------------------###
from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

class Missing(ModelPipline):
    def __init__(self,
                 x_train,
                 y_train=None,
                 flag=None,
                 plot=False,
                 cols=None,
                 stats_name="mean",
                 x_test=None,
                 feature_cols=None,
                 knn_num=2,
                 max_iter=5):
        """
        x_train, y_trainはPandas.DataFrameであること
        """
        self.x_train = x_train 
        self.y_train = y_train 
        self.plot = plot
        self.cols = cols
        self.stats = stats_name
        self.x_test = x_test
        self.feature_cols = feature_cols
        self.knn_num = knn_num
        self.max_iter = max_iter
        
        if flag == "tree":
            self.tree()
        elif flag == "dropna":
            self.drop_value()
        elif flag == "stats":
            self.stats_value()
        elif flag == "linear":
            self.Regression_completion()
        elif flag == "knn":
            self.KNN_completion()
#         elif flag == "multi":
#             self.Multiple_imputation()
        else:
            print("flag引数を指定してください。")
        
    def tree(self):
        """
        欠損値を-9999で保管して決定木で評価してみます
        意図的な欠損とみられる、欠損することで目的変数との相関がみられる場合に有効です
        
        x_train
        y_train
        flag: tree
        """
        print("欠損値を-9999で変換して、評価を行います")
        x_train = self.x_train.fillna(-9999)
        y_train = self.y_train 
        ModelPipline(x_train, y_train)
        
    
    def drop_value(self):
        """
        欠損がある行そのものを削除します
        全データのうち、数パーセントまたはデータのほとんどが欠損である場合に有効です
        
        x_train: 訓練データ
        y_train: 訓練データの正解ラベル
        flag: dropna
        plot: 処理後にモデルの評価を行うか boolean
        """
        x_train = self.x_train
        y_train = self.y_train
        len_x = x_train.shape[0]
        datasets = pd.concat([x_train, y_train], axis=1).reset_index().drop(["index"], axis=1)
        datasets = datasets.dropna()
        x_train, y_train = datasets.iloc[:, :-1], datasets.iloc[:, [-1]]
        len_x_trans = x_train.shape[0]
        if self.plot:
            #特長量を落としたことによるモデルの評価の確認
            ModelPipline(x_train, y_train)
        else:
            print(f"{len_x} から {len_x_trans} に変換しました")
            x_train.to_csv("./transform/x_train_dropna.csv", index=False)
        
    def stats_value(self):
        """
        欠損値を代表値で保管します
        
        x_train: 訓練データ
        stats_name: 代表値の文字列
        cols: 欠損しているカラム名,複数の設定も可能
        test: テストデータ
        flag: stats
        """
        if self.stats != None:
            x_train_copy = self.x_train.copy()
            x_test_copy = self.x_test.copy()
            imputer = SimpleImputer(missing_values=np.nan, strategy=self.stats)
            missing_cols = self.cols
            imputer.fit(self.x_train[missing_cols])
            x_train_copy[[c + "_" + str(self.stats) for c in missing_cols]] = pd.DataFrame(
                imputer.transform(self.x_train[missing_cols]),
                index=self.x_train.index
            )
            x_test_copy[[c + "_" + str(self.stats) for c in missing_cols]] = pd.DataFrame(
                imputer.transform(self.test[missing_cols]),
                index=self.test.index
            )
            print(f"{missing_cols}を{self.stats}で補完します。")
            
            x_train_copy.to_csv(f"./transform/x_train_stats_storage.csv", index=False)
            x_test_copy.to_csv(f"./transform/x_test_stats_storage.csv", index=False)
            
        
        else:
            print("引数colsにカラム名を、引数stats_nameにmean, median, most_frequentを指定してください。")
            
    def Regression_completion(self):
        """
        欠損値を予測補完します
        1. 回帰
        2. k-nn
        3. 多重代入法
        
        
        x_train: 訓練データの正解ラベルは含めません
        flag: linear
        x_test 
        cols: 欠損補完したいカラム名のリスト型、複数指定可能
        feature_cols: (任意)予測時に用いたくないカラム名のリスト型 == 欠損補完するデータ群との相関がみられないカラム == 欠損しているカラム名(予測に使えないため)
        """
        x_train_copy = self.x_train.copy()
        x_test_copy = self.x_test.copy()
        for col in self.cols:
            reg = LinearRegression()
            #訓練と検証データの組み合わせでモデリング
            dataset = pd.concat([self.x_train, self.x_test], axis=0).reset_index().drop(["index"], axis=1)
            ix = dataset[col].isnull()

            #回帰に使う特長量の選択,ここでは欠損していないデータを用いる
            if self.feature_cols != None:
                x1 = dataset[~ix].drop(self.feature_cols, axis=1)
                x1 = x1.dropna(axis=1)
                x1_col = list(x1.columns)
                train_reg = self.x_train[self.x_train[col].isnull()].drop(self.feature_cols, axis=1)
                train_reg = train_reg[x1_col]
                test_reg = self.x_test[self.x_test[col].isnull()].drop(self.feature_cols, axis=1)
                test_reg = test_reg[x1_col]

            else:
                x1 = dataset[~ix]
                x1 = x1.dropna(axis=1)
                x1_col = list(x1.columns)
                train_reg = self.x_train[self.x_train[col].isnull()]
                train_reg = train_reg[x1_col]
                test_reg = self.x_test[self.x_test[col].isnull()]
                test_reg = test_reg[x1_col]
                
            #欠損していない正解ラベルの作成
            x2 = dataset.loc[x1.index, col]
            x1 = x1.drop([col], axis=1)
            scaler = StandardScaler()
            x1 = scaler.fit_transform(x1)
            reg.fit(x1, x2.values)
            
            
            train_reg = train_reg.drop([col], axis=1)
            scaler = StandardScaler()
            scaler.fit(train_reg)
            train_re = scaler.transform(train_reg)
            test_reg = test_reg.drop([col], axis=1)
            test_re = scaler.transform(test_reg)
            
            x_train_copy[col + "_" + "reg"] = pd.Series(
                reg.predict(train_re),
                train_reg.index
            ) 
            x_train_copy[col + "_" + "reg"] =  x_train_copy[col + "_" + "reg"].fillna(x_train_copy[col])
            
            x_test_copy[col + "_" + "reg"] = pd.Series(
                reg.predict(test_re),
                test_reg.index
            )
            x_test_copy[col + "_" + "reg"] =  x_test_copy[col + "_" + "reg"].fillna(x_test_copy[col])


        x_train_copy.to_csv("./transform/x_train_regression_completion.csv", index=False)
        x_test_copy.to_csv("./transform/x_test_regression_completion.csv", index=False)
        
        #元データと予測補完によるデータの差異を可視化します
        plt.figure(figsize=(14, 7))
        for i, col in enumerate(self.cols):
            plt.subplot(2, 2, i+1)
            plt.hist(x_train_copy[col], alpha=0.7, label="complete data")
            plt.legend()
            plt.hist(x_train_copy[col+"_"+"reg"], alpha=0.7)
            plt.grid()
            plt.title(col)

            
    def KNN_completion(self):
        """
        回帰と同じ引数指定

        knn_num: KNNのパラメータ
        flag: knn
        """
        x_train_copy = self.x_train.copy()
        x_test_copy = self.x_test.copy()
        for col in self.cols:
        
            #データ間の隣人指定の設定
            imputer = KNNImputer(n_neighbors=self.knn_num)

            if self.feature_cols != None:
                imputer.fit(self.x_train.drop(self.feature_cols, axis=1))
                
                x_train_copy[col + "_" + "knn"] = pd.DataFrame(
                    imputer.transform(self.x_train.drop(self.feature_cols, axis=1)),
                    index=self.x_train.drop(self.feature_cols, axis=1).index,
                    columns=self.x_train.drop(self.feature_cols, axis=1).columns
                )[col]
                
                x_test_copy[col + "_" + "knn"] = pd.DataFrame(
                    imputer.transform(self.x_test.drop(self.feature_cols, axis=1)),
                    index=self.x_test.drop(self.feature_cols, axis=1).index,
                    columns=self.x_test.drop(self.feature_cols, axis=1).columns
                )[col]
            else:
                imputer.fit(self.x_train)
                
                x_train_copy[col + "_" + "knn"] = pd.DataFrame(
                    imputer.transform(self.x_train),
                    index=self.x_train.index,
                    columns=self.x_train.columns
                )[col]
                
                x_test_copy[col + "_" + "knn"] = pd.DataFrame(
                    imputer.transform(self.x_test),
                    index=self.x_test.index,
                    columns=self.x_test.columns
                )[col]
                
        
        x_train_copy.to_csv("./transform/x_train_knn_completion.csv", index=False)
        x_test_copy.to_csv("./transform/x_test_knn_completion.csv", index=False)
        
        plt.figure(figsize=(14, 7))

        for i, col in enumerate(self.cols):
            plt.subplot(2, 2, i+1)
            plt.hist(x_train_copy[col], alpha=0.7, label="complete data")
            plt.legend()
            plt.hist(x_train_copy[col+"_"+"knn"], alpha=0.7)
            plt.grid()
            plt.title(col)
    
#     def Multiple_imputation(self):
#         """
#         多重代入法による補完を行います
        
#         引数は回帰と同じ flag: multi
#         max_iter: パラメータ指定int型
#         """
#         for col in self.cols:
        
#             #データ間の隣人指定の設定
#             imputer = IterativeImputer(max_iter=self.max_iter, sample_posterior=True, random_state=0)
#             x_train_copy = self.x_train.copy()
#             x_test_copy = self.x_test.copy()

#             if self.feature_cols != None:
#                 imputer.fit(self.x_train.drop(self.feature_cols, axis=1))
                
#                 x_train_copy[col + "_" + "knn"] = pd.DataFrame(
#                     imputer.transform(self.x_train.drop(self.feature_cols, axis=1)),
#                     index=self.x_train.drop(self.feature_cols, axis=1).index,
#                     columns=self.x_train.drop(self.feature_cols, axis=1).columns
#                 )[col]
                
#                 x_test_copy[col + "_" + "knn"] = pd.DataFrame(
#                     imputer.transform(self.x_test.drop(self.feature_cols, axis=1)),
#                     index=self.x_test.drop(self.feature_cols, axis=1).index,
#                     columns=self.x_test.drop(self.feature_cols, axis=1).columns
#                 )[col]
#             else:
#                 imputer.fit(self.x_train)
                
#                 x_train_copy[col + "_" + "knn"] = pd.DataFrame(
#                     imputer.transform(self.x_train),
#                     index=self.x_train.index,
#                     columns=self.x_train.columns
#                 )[col]
                
#                 x_test_copy[col + "_" + "knn"] = pd.DataFrame(
#                     imputer.transform(self.x_test),
#                     index=self.x_test.index,
#                     columns=self.x_test.columns
#                 )[col]
                
        
#         x_train_copy.to_csv("./transform/x_train_knn_completion.csv", index=False)
#         x_test_copy.to_csv("./transform/x_test_knn_completion.csv", index=False)
        
#         plt.figure(figsize=(14, 7))

#         for i, col in enumerate(self.cols):
#             plt.subplot(2, 2, i+1)
#             plt.hist(x_train_copy[col], alpha=0.7, label="complete data")
#             plt.legend()
#             plt.hist(x_train_copy[col+"_"+"knn"], alpha=0.7)
#             plt.grid()
#             plt.title(col)

###------------------------------数値変換--------------------------------------------###
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
import numpy as np 
import pandas as pd 

class Continuous(ModelPipline):
    def __init__(self, 
                 x_train,
                 x_test,
                 x_val=pd.DataFrame(),
                 y_train=pd.DataFrame(),
                 standard=[], 
                 minmax=[], 
                 log=[],
                 log1p=[], 
                 boxcox=[], 
                 yao=[],
                 clipping=[], 
                 binning=[], 
                 n_bins=4,
                 rank=[]):
        # data frame
        self.x_train = x_train 
        self.y_train = y_train 
        self.x_val = x_val 
        self.x_test = x_test
        # transform methods 
        self.standard = standard 
        self.minmax = minmax 
        self.log = log 
        self.log1p = log1p
        self.boxcox = boxcox
        self.yao = yao 
        self.clipping = clipping
        self.binning = binning 
        self.n_bins = n_bins 
        self.rank = rank
        
        self.trans()
    def trans(self):
        """
        x_train:
        x_test:
        columns: List
        
        y_train: DataFrame
        """
        
        #標準化
        if len(self.standard) > 0:
            for col in self.standard:
                scaler = StandardScaler()
                scaler.fit(self.x_train[col].values.reshape(-1, 1))
                self.x_train[col] = scaler.transform(self.x_train[col].values.reshape(-1, 1))
                self.x_test[col] = scaler.transform(self.x_test[col].values.reshape(-1, 1))
                if self.x_val.shape[0] > 0:
                    self.x_val[col] = scaler.transform(self.x_val[col].values.reshape(-1, 1))
        #最大最小        
        if len(self.minmax) > 0:
            for col in self.minmax:
                scaler = MinMaxScaler()
                scaler.fit(self.x_train[col].values.reshape(-1, 1))
                self.x_train[col] = scaler.transform(self.x_train[col].values.reshape(-1, 1))
                self.x_test[col] = scaler.transform(self.x_test[col].values.reshape(-1, 1))
                if self.x_val.shape[0] > 0:
                    self.x_val[col] = scaler.transform(self.x_val[col].values.reshape(-1, 1))
        #対数        
        if len(self.log) > 0:
            for col in self.log:
                x = self.x_train[col].values
                t = self.x_test[col].values
                x = np.sign(x) * np.log(np.abs(x))
                t = np.sign(t) * np.log(np.abs(t))
                self.x_train[col] = x 
                self.x_test[col] = t
                if self.x_val.shape[0] > 0:
                    x = self.x_val[col].values
                    t = np.sign(x) * np.log(np.abs(x))
                    self.x_val[col] = t
                
        #負の値を含む対数        
        if len(self.log1p) > 0:
            for col in self.log1p:
                x = self.x_train[col].values
                t = self.x_test[col].values
                x = np.log1p(x)
                t = np.log1p(t)
                self.x_train[col] = x 
                self.x_test[col] = t
                if self.x_val.shape[0] > 0:
                    x = self.x_val[col].values
                    t = np.sign(x) * np.log(np.abs(x))
                    self.x_val[col] = t
                
        #正規分布化        
        if len(self.boxcox) > 0:
            for col in self.boxcox:
                pt = PowerTransformer(method="box-cox")
                pt.fit(self.x_train[col])
                self.x_train[col] = pt.transform(self.x_train[col])
                self.x_test[col] = pt.transform(self.x_test[col])
                if self.x_val.shape[0] > 0:
                    self.x_val[col] = pt.transform(self.x_val[col])
                
        #負の値の正規分布化    
        if len(self.yao) > 0:
            for col in self.boxcox:
                pt = PowerTransformer(method="yao-johnson")
                pt.fit(self.x_train[col])
                self.x_train[col] = pt.transform(self.x_train[col])
                self.x_test[col] = pt.transform(self.x_test[col])
                if self.x_val.shape[0] > 0:
                    self.x_val[col] = pt.transform(self.x_val[col])
                    
        #上限下限変換        
        if len(self.clipping) > 0:
            for col in self.clipping:
                p01 = self.x_train[col].quantile(.01)
                p99 = self.x_train[col].quantile(.99)
                self.x_train[col] = self.x_train[col].clip(p01, p99, axis=1)
                self.x_test[col] = self.x_test[col].clip(p01, p99, axis=1)
                if self.x_val.shape[0] > 0:
                    self.x_val[col] = self.x_val[col].clip(p01, p99, axis=1)
                    
        #順位尺度変換        
        if len(self.binning) > 0:
            for col in self.binning:
                self.x_train[col+"_binning"] = pd.cut(self.x_train[col], self.n_bins, labels=False)
                self.x_test[col+"_binning"] = pd.cut(self.x_test[col], self.n_bins, labels=False)
                if self.x_val.shape[0] > 0:
                    self.x_val[col+"_binning"] = pd.cut(self.x_val[col], self.n_bins, labels=False)
        #ランキングの正規分布化
        if len(self.rank) > 0:
            for col in self.rank:
                q = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
                q.fit(self.x_train[col])
                self.x_train[col+"_rank"] = q.transform(self.x_train[col])
                self.x_test[col+"_rank"] = q.transform(self.x_test[col])
                if self.x_val.shape[0] > 0:
                    self.x_val[col+"_rank"] = q.transform(self.x_val[col])
                
        try:
            if self.y_train.shape[1] > 0:
                ModelPipline(self.x_train, self.y_train)
        except:
            print("モデルの検証に失敗しました。")
        finally:
            print("変換後のファイルの出力を行います.....")
            self.x_train.to_csv("./transform/x_train_transcontinus.csv", index=False)
            self.x_test.to_csv("./transform/x_test_transcontinus.csv", index=False)
            if self.x_val.shape[0] > 0:
                self.x_val.to_csv("./transform/x_val_transcontius.csv", index=False)

###-----------------------------カテゴリー変換--------------------------------------------###
# !pip install -q category_encoders
import category_encoders as ce 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import pandas as pd 

class Categorical(ModelPipline):
    def __init__(self, 
                 x_train,
                 x_test,
                 x_val=pd.DataFrame(),
                 y_val=pd.DataFrame(), 
                 y_train=pd.DataFrame(),
                 onehot=[], 
                 label=[],
                 frequency=[], 
                 target=[],
                 n_fold=4,
                 method="mean"):
        self.x_train = x_train 
        self.x_test = x_test 
        self.x_val = val #ターゲットエンコーダーにのみ使います。リークを防ぐため
        self.y_train = y_train 
        self.y_val =y_val
        
        self.onehot = onehot 
        self.label = label
        self.frequency = frequency
        self.target = target 
        self.n_fold = n_fold
        self.method = method 
        
        self.trans()
    def trans(self): 
        # One-Hot-encoder
        if len(self.onehot) > 0:
            for col in self.onehot:
                encoder = ce.OneHotEncoder(use_cat_names=True)
                encoder.fit(self.x_train[col])
                train = encoder.transform(self.x_train[col])
                train = train.drop([c for c in train.columns if c.find(col+"_") > -1][0], axis=1) # drop first 
                train = pd.merge(self.x_train.drop([col], axis=1), train, how="left", left_index=True, right_index=True)
                test = encoder.transform(self.x_test[col])
                test = test.drop([c for c in test.columns if c.find(col+"_") > -1][0], axis=1)
                test = pd.merge(self.x_test.drop([col], axis=1), test, how="left", left_index=True, right_index=True)
                
                self.x_train = train 
                self.x_test = test 
                
        # Label-encoder        
        if len(self.label) > 0:
            for col in self.label:
                la = LabelEncoder()
                la.fit(self.x_train[col])
                self.x_train[col] = la.transform(self.x_train[col])
                self.x_test[col] = la.transform(self.x_test[col])
        # frequency-encoder     
        if len(self.frequency) > 0:
            for col in self.frequency:
                freq = self.x_train[col].value_counts()
                self.x_train[col] = self.x_train[col].map(freq)
                self.x_test[col] = self.x_test[col].map(freq)
        # Target-encoder
        if len(self.target) > 0:
            if self.method == "mean":
                for col in self.target:
                    data_tmp = pd.DataFrame({col: self.x_train[col], "target": self.y_train.values.ravel()})
                    target_mean = data_tmp.groupby(col)["target"].mean()
                    if self.x_val.shape[0] > 0:
                        self.x_val[col] = self.x_val[col].map(target_mean)
                    self.x_test[col] = self.x_test[col].map(target_mean)

                    tmp = np.repeat(np.nan, self.x_train.shape[0])
                    print(f"{col}へのターゲットエンコーダーを行います。なお、検証データは訓練yの{self.method}を用います。訓練データ分割{self.n_fold}回で予測変換します。リークに注意してください。")
                    
                    kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=0)
                    for id1, id2 in kf.split(self.x_train):

                        target_mean = data_tmp.iloc[id1].groupby(col)["target"].mean()
                        tmp[id2] = self.x_train[col].iloc[id2].map(target_mean)

                    self.x_train[col] = tmp 
            elif self.method == "median":
                for col in self.target:
                    data_tmp = pd.DataFrame({col: self.x_train[col], "target": self.y_train.values.ravel()})
                    target_mean = data_tmp.groupby(col)["target"].median()
                    if self.x_val.shape[0] > 0:
                        self.x_val[col] = self.x_val[col].map(target_mean)
                    self.x_test[col] = self.x_test[col].map(target_mean)

                    tmp = np.repeat(np.nan, self.x_train.shape[0])
                    print(f"{col}へのターゲットエンコーダーを行います。なお、検証データは訓練yの{self.method}を用います。訓練データ分割{self.n_fold}回で予測変換します。リークに注意してください。")

                    kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=0)
                    for id1, id2 in kf.split(self.x_train):

                        target_mean = data_tmp.iloc[id1].groupby(col)["target"].median()
                        tmp[id2] = self.x_train[col].iloc[id2].map(target_mean)

                    self.x_train[col] = tmp 
                    
        try:
            
            if (self.y_train.shape[1] > 0) and (self.x_val.shape[0] > 0) and (self.y_val.shape[0] > 0):
                ModelPipline(self.x_train, self.y_train, self.x_val, self.y_val)
            elif self.y_train.shape[1] > 0:
                ModelPipline(self.x_train, self.y_train)
                
        except:
            print("モデルの検証に失敗しました。")
        
        finally:
            print("ファイルの出力を行います")       
            self.x_train.to_csv("./transform/x_train_category.csv", index=False)
            self.x_test.to_csv("./transform/x_test_category.csv", index=False)
            if self.x_val.shape[0] > 0:
                self.x_val.to_csv("./transform/x_val_category.csv", index=False)

                
