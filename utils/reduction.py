from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import MiniBatchKMeans
import plotly.express as px 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#!pip install umap-learn
# import umap

###---------------------------------------次元削減------------------------------------------------------###
class Reduction:
    def __init__(self, x_train, y_train, x_test, x_val=pd.DataFrame([])):
        self.x_train = x_train 
        self.y_train = y_train 
        self.x_val = x_val
        self.x_test = x_test 
    
        
    def PCA(self, n_components=2, plot=True):
        """
        pcaによる次元削減を行います
        
        x_train: 
        y_train: 
        x_test:
        plot: boolean 
        n_components: int 
        """
        if plot:
            pca = PCA(n_components=2)
            x_train_pca = pca.fit_transform(self.x_train)
            fig = px.scatter(x=x_train_pca[:, 0], 
                             y=x_train_pca[:, 1],
                             color=self.y_train.values.ravel(),
                            title="PCA")
            fig.show()
            th = pca.explained_variance_ratio_
            print(f"第一寄与率: {th[0]} 第二寄与率 : {th[1]}")
            
        
        else:
            pca = PCA(n_components=n_components)
            pca.fit(self.x_train)
            x_train_pca = pca.transform(self.x_train)
            x_train_pca = pd.DataFrame(x_train_pca)
            if self.x_val.shape[0] > 0:
                x_val_pca = pca.transform(self.x_val)
                x_val_pca = pd.DataFrame(x_val_pca)
            x_test_pca = pca.transform(self.x_test)
            x_test_pca = pd.DataFrame(x_test_pca)

            
            x_train_pca.columns = [str(c+1)+"pricipal" for c in range(n_components)]
            if self.x_val.shape[0] > 0:
                x_val_pca.columns = [str(c+1)+"pricipal" for c in range(n_components)]
            x_test_pca.columns = [str(c+1)+"principal" for c in range(n_components)]
            
            ratio = pca.explained_variance_ratio_.cumsum()
            ratio = np.concatenate([np.array([0]), ratio])

            plt.plot(ratio)
            plt.xticks(np.arange(len(ratio)))
            plt.ylabel('explained_variance_ratio_')
            plt.xlabel('pca-number')
            plt.title('PCA')
            plt.tight_layout()
            
            if self.x_val.shape[0] > 0:
                return x_train_pca, x_val_pca, x_test_pca 
            else:
                return x_train_pca, x_test_pca 
            
    def LDA(self, n_components=1):
        """
        教師ありで次元削減を行う線形分析手法/分類タスクに限定される
        n_componentsには、分類クラスよりも小さい値とする
        
        x_train:
        y_train: 
        x_test:
        """
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        
        lda.fit(self.x_train, self.y_train)
        x_train_lda = lda.transform(self.x_train)
        x_train_lda = pd.DataFrame(x_train_lda, columns=[str(c)+"_lda" for c in range(n_components)])
        if self.x_val.shape[0] > 0:
            x_val_lda = lda.transform(self.x_val)
            x_val_lda = pd.DataFrame(x_val_lda, columns=[str(c)+"_lda" for c in range(n_components)])
   
        x_test_lda = lda.transform(self.x_test)
        x_test_lda = pd.DataFrame(x_test_lda, columns=[str(c)+"_lda" for c in range(n_components)])
        
        if self.x_val.shape[0] > 0:
            return x_train_lda, x_val_lda, x_test_lda
        else:
            return x_train_lda, x_test_lda 
        

    def TSNE(self, plot_2d=True, n_components=2, perplexity=25):
        """
        非線形による次元削減
        計算コストが高く3次元を超える次元圧縮には不向きです。
        
        x_train 
        y_train:
        n_coponents: 変換後の次元数
        perplexity: tsneオプション
        plot_2d: 2次元での可視化
        """
        if plot_2d:
            for i in [5, 25, 30, 50]:
                tsne = TSNE(n_components=2, perplexity=i, random_state=0).fit_transform(self.x_train)

                fig = px.scatter(x=tsne[:, 0], y=tsne[:, 1], color=self.y_train.values.ravel(), title=f"TSNE perplexity={i}:")
                fig.show()
        else:
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=0)
            x_train_t = tsne.fit_transform(self.x_train)
            x_test_t = tsne.fit_transform(self.x_test)
            if self.x_val.shape[0] > 0:
                x_val_t = tsne.fit_transform(self.x_val)
                
            x_train_c = self.x_train.copy()
            x_val_c = self.x_val.copy()
            x_test_c = self.x_test.copy()
            
            for i in range(n_components):
                x_train_c[f"{i}_tsne"] = x_train_t[:, i]
                x_test_c[f"{i}_tsne"] = x_test_t[:, i]
                if self.x_val.shape[0] > 0:
                    x_val_c[f"{i}_tsne"] = x_val_t[:, i]
            
            if self.x_val.shape[0] > 0:
                return x_train_c, x_val_c, x_test_c 
            else:
                return x_train_c, x_test_c
#     def Umap(self):
#         """
#         非線形による次元削減
        
#         x_train:
#         y_train:
#         """
        
#         um = umap.UMAP(n_components=2)
#         x_train_um = um.fit_transform(self.x_train)
        
#         fig = px.scatter(x=x_train_um[:, 0], y=x_train_um[:, 1], color=self.y_train, title="umap")
#         fig.show()
        
        
    def MDS(self, plot_2d=True, n_components=2):
        """
        ストレス値を基準にして元データ空間と変換データ空間の誤差を求めます
        最初にself.plotを実行して最適なstressの可視化を行いましょう
        次にself.plot=Falseでn_componentsに最適なstreeeになる次元数を指定して.csvに書き出し実行しましょう
        
        x_train:
        plot: boolean
        x_test:
        n_component: int 
        """
        if plot_2d:
            l =[]
            for i in np.arange(len(self.x_train.columns)):
                embedding = MDS(n_components=i+1, metric=True, random_state=1)
                embedding.fit_transform(self.x_train)
                l.append(embedding.stress_)
                
            print("stress値が0に近いほど削減性が高まります。")
                
            plt.plot(np.arange(len(self.x_train.columns))+1, l)
            plt.xlabel("Number of dimensions")
            plt.ylabel("stress")
            plt.xlim(0, len(self.x_train.columns))
            plt.xticks(np.arange(len(self.x_train.columns))+1)
            plt.show()
            
        else:
            embedding = MDS(n_components=n_components, random_state=1)
            x_train_mds = embedding.fit_transform(self.x_train)
            x_train_mds = pd.DataFrame(x_train_mds, columns=[str(i+1) + "principal" for i in range(n_components)])
            x_test_mds = embedding.fit_transform(self.x_test)
            x_test_mds = pd.DataFrame(x_test_mds, columns=[str(i+1) + "principal" for i in range(n_components)])
            
            if self.x_val.shape[0] > 0:
                x_val_mds = embedding.fit_transform(self.x_val)
                x_val_mds = pd.DataFrame(x_val_mds, columns=[str(i+1) + "principal" for i in range(n_components)])
            
            print(f"多次元尺度構成により、{n_components}次元に削減しました。")
            
            if self.x_val.shape[0] > 0:
                return x_train_mds, x_val_mds, x_test_mds 
            else:
                return x_train_mds, x_test_mds 

    def cluster(self, n_components):
        """
        教師なしによるクラスたリングを行い、分類後の中心点からの距離を新たな特長量とします
        
        x_train:
        x_test:
        n_components: クラスタ数
        """
        kmeans = MiniBatchKMeans(n_clusters=n_components, random_state=0)
        kmeans.fit(self.x_train)
        
        train_cluster = kmeans.predict(self.x_train)
        if self.x_val.shape[0] > 0:
            val_cluster = kmeans.predict(self.x_val)
        test_cluster = kmeans.predict(self.x_test)
        
        train = self.x_train.copy()
        val = self.x_val.copy()
        test = self.x_test.copy()
        
        train_distance = kmeans.transform(self.x_train)
        train_distance_min = train_distance.min(axis=1)
        if self.x_val.shape[0] > 0:
            val_distance = kmeans.transform(self.x_val)
            val_distance_min = val_distance.min(axis=1)
        test_distance = kmeans.transform(self.x_test)
        test_distance_min = test_distance.min(axis=1)
        
        train["cluster"] = train_cluster
        test["cluster"] = test_cluster
        train["kmeans_distance_min"] = train_distance_min.reshape(-1, 1)
        test["kmeans_distance_min"] = test_distance_min.reshape(-1, 1)
        if self.x_val.shape[0] > 0:
            val["cluster"] = val_cluster 
            val["kmeans_distance_min"] = val_distance_min.reshape(-1, 1)
    
        for i in range(n_components):
            train[f"kmeans_distance_{i+1}"] = train_distance[:, i]
            test[f"kmeans_distance_{i+1}"] = test_distance[:, i]
            if self.x_val.shape[0] > 0:
                val[f"kmeans_distance_{i+1}"] = val_distance[:, i]
        
        print(f"clusterを{n_components}に分割、重心からの距離をカラムに追加しました。")
        if self.x_val.shape[0] > 0:
            return train, val, test
        else:
            return train, test 
            

###-----------------------------------------特徴量選択----------------------------------------------------------###
        
# !pip install -q mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, SelectFromModel
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px 

#特長量の削減をそれぞれの手法を用いて行います
class SelectFeature:
    def __init__(self,
                 x_train,
                 y_train, 
                 x_test,
                 x_val=pd.DataFrame(),
                 y_val=pd.DataFrame()): # 指定したいvalidation-datasetがあれば入力/なければランダム分割による検証スコアを参照する
        
        self.x_train = x_train 
        self.y_train = y_train
        # 必ずx_val, y_valはセットで入力すること
        self.x_val = x_val 
        self.y_val = y_val
        self.x_test = x_test 

            
    def fit_filter(self):
        """
        統計を基に最適な特長量の選択を行います。
        
        x_train:
        y_train:
        x_val:
        y_val
        x_test:
        """
        
        tr_score = []
        pre_val_score = []
        after_val_score = []
        features = []
        
        if self.x_val.shape[0] > 0:
            for i in np.arange(len(self.x_train.columns)):

                if len(self.y_train[self.y_train.columns.item()].unique()) < 10:
                    sel = SelectKBest(f_classif, k=i+1).fit(self.x_train, self.y_train)
                    x_train_sel = sel.transform(x_train)
                    x_val_sel = sel.transform(x_val)
                    log_pre = LogisticRegression().fit(self.x_train, self.y_train)
                    log = LogisticRegression().fit(x_train_sel, y_train)
                    tr_score.append(log.score(x_train_sel, y_train))
                    pre_val_score.append(log_pre.score(self.x_val, self.y_val))
                    after_val_score.append(log.score(x_val_sel, y_val))
                    features.append(self.x_train.columns[sel.get_support()])
                else:
                    sel = SelectKBest(f_regression, k=i+1).fit(self.x_train, self.y_train)
                    x_train_sel = sel.transform(x_train)
                    x_val_sel = sel.transform(x_val)
                    log_pre = LinearRegression().fit(self.x_train, self.y_train)
                    log = LinearRegression().fit(x_train_sel, y_train)
                    tr_score.append(log.score(x_train_sel, y_train))
                    pre_val_score.append(log_pre.score(self.x_val, self.y_val))
                    after_val_score.append(log.score(x_val_sel, y_val))
                    features.append(self.x_train.columns[sel.get_support()])
        else:
        
            x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, random_state=0)

            for i in np.arange(len(self.x_train.columns)):

                if len(self.y_train[self.y_train.columns.item()].unique()) < 10:
                    sel = SelectKBest(f_classif, k=i+1).fit(self.x_train, self.y_train)
                    x_train_sel = sel.transform(x_train)
                    x_val_sel = sel.transform(x_val)
                    log_pre = LogisticRegression().fit(x_train, y_train)
                    log = LogisticRegression().fit(x_train_sel, y_train)
                    tr_score.append(log.score(x_train_sel, y_train))
                    pre_val_score.append(log_pre.score(x_val, y_val))
                    after_val_score.append(log.score(x_val_sel, y_val))
                    features.append(self.x_train.columns[sel.get_support()])
                else:
                    sel = SelectKBest(f_regression, k=i+1).fit(self.x_train, self.y_train)
                    x_train_sel = sel.transform(x_train)
                    x_val_sel = sel.transform(x_val)
                    log_pre = LinearRegressionar.fit(x_train, y_train)                   
                    log = LinearRegression().fit(x_train_sel, y_train)
                    tr_score.append(log.score(x_train_sel, y_train))
                    pre_val_score.append(log_pre.score(x_val, y_val))
                    after_val_score.append(log.score(x_val_sel, y_val))
                    features.append(self.x_train.columns[sel.get_support()])
                
        feature_sel_filter = features[tr_score.index(np.max(tr_score))]
        
        print(f"特長量 : {feature_sel_filter}")
        print(f"訓練 : {tr_score[tr_score.index(np.max(tr_score))]}")
        print(f"検証 : {after_val_score[tr_score.index(np.max(tr_score))]}")
        
        print(f"{len(self.x_train.columns)}から{len(feature_sel_filter)}に次元削減しました。")
#         self.x_train[feature_sel_filter].to_csv("./transform/x_train_filter.csv", index=False)
#         self.x_test[feature_sel_filter].to_csv("./transform/x_test_filter.csv", index=False)
        
        plt.figure(figsize=(14, 7))
        plt.plot(np.arange(len(self.x_train.columns))+1, pre_val_score, label="before_validation_score")
        plt.plot(np.arange(len(self.x_train.columns))+1, after_val_score, label="after_validation_score")
        plt.ylabel("Accuracy")
        plt.xlabel("Number of features")
        plt.xticks(np.arange(1, 1+len(self.x_train.columns)))
        plt.legend()
        plt.grid()
        plt.title("feature selection by using F values")
        
        if self.x_val.shape[0] > 0:
            return self.x_train[feature_sel_filter], self.x_val[feature_sel_filter], self.x_test[feature_sel_filter]
        else:
            return self.x_train[feature_sel_filter], self.x_test[feature_sel_filter]
        
    def fit_wrapper(self, floating=False, verbose=2, scoring="accuracy", forward=True, cv=5):
        """
        一つ一つ試しながら検証することで最適な特長量選択を行う。
        
        x_train:
        y_train:
        x_val: 検証データが事前にあれば定義する
        y_val:
        x_test:
        """
        if len(self.y_train[self.y_train.columns.item()].unique()) < 10:
            sfs = SFS(LogisticRegression(),
                     k_features=(1, len(self.x_train.columns)),
                     floating=floating,
                     forward=forward,
                     verbose=verbose,
                     scoring=scoring,
                     cv=cv,
                     n_jobs=-1)
         
            sfs = sfs.fit(self.x_train, self.y_train)
 
            
            features_sel_wrapper = list(sfs.k_feature_names_)
            print(f"特長量 : {features_sel_wrapper}")
            
            if self.x_val.shape[0] > 0:
                x_train = self.x_train
                y_train = self.y_train
                x_val = self.x_val
                y_val = self.y_val 
            else:
                x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, random_state=0)
            ll = LogisticRegression().fit(x_train, y_train)
            l = LogisticRegression().fit(x_train[features_sel_wrapper], y_train)
            print(f"変換前検証 : {ll.score(x_val, y_val)}")
            print(f"変換後検証 : {l.score(x_val[features_sel_wrapper], y_val)}")
            
        else:
            sfs = SFS(LinearRegression(),
                     k_features=(1, len(self.x_train.columns)),
                     floating=floating,
                     forward=forward,
                     verbose=verbose,
                     scoring=scoring,
                     cv=cv,
                     n_jobs=-1)
            sfs = sfs.fit(self.x_train, self.y_train)
            
            features_sel_wrapper = list(sfs.k_feature_names_)
            print(f"特長量 : {features_sel_wrapper}")
            if self.x_val.shape[0] > 0:
                x_train = self.x_train
                y_train = self.y_train
                x_val = self.x_val
                y_val = self.y_val 
            else:
                x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, random_state=0)
            ll = LinearRegression().fit(x_train, y_train)
            l = LinearRegression().fit(x_train[features_sel_wrapper], y_train)
            print(f"変換前検証 : {ll.score(x_val, y_val)}")
            print(f"変換後検証 : {l.score(x_val[features_sel_wrapper], y_val)}")
            

        print(f"{len(self.x_train.columns)}から{len(features_sel_wrapper)}に次元削減しました。")
        
        fig = plot_sfs(sfs.get_metric_dict(), kind="std_dev", figsize=(14, 7))
        plt.title("Sequence Backward Generation")
        plt.grid()
        
        if self.x_val.shape[0] > 0:
            return self.x_train[features_sel_wrapper], self.x_val[features_sel_wrapper], self.x_test[features_sel_wrapper]
        else:
            return self.x_train[features_sel_wrapper], self.x_test[features_sel_wrapper]
            
            
    def fit_embedded(self, method="tree", value="mean"):
        """
        ランダム決定木、Lasso回帰による特長量の選択を行います。
        具体的には、モデルにおける重みパラメータが平均を下回っている特長の削減します。
        
        x_train: 
        y_train:
        x_val:
        y_val:
        x_test:
        """
        
        if self.x_val.shape[0] > 0:
            
            x_train, x_val, y_train, y_val = self.x_train, self.x_val, self.y_train, self.y_val
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

            if method == "tree":

                embeded_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0, min_samples_leaf=50), value)
                embeded_selector.fit(self.x_train, self.y_train)

                feature_sel_embed = self.x_train.columns[embeded_selector.get_support()]
                re = LogisticRegression().fit(x_train, y_train)
                reg = LogisticRegression().fit(embeded_selector.transform(x_train), y_train)
                print(f"特長量 : {feature_sel_embed}")

                print(f"決定木による特長選択で{len(self.x_train.columns)}から{len(feature_sel_embed)}に次元削減しました。")
    
                try:

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_train, y_train), reg.score(embeded_selector.transform(x_train), y_train)], 
                                 title="train metrics")
                    fig.show()

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_val, y_val), reg.score(embeded_selector.transform(x_val), y_val)], 
                                 title="validation metrics")
                    fig.show()
                except:
                    print("not modules ploly")
                finally:
                
                    return self.x_train[feature_sel_embed], self.x_val[feature_sel_embed], self.x_test[feature_sel_embed]

            else:
                embeded_selector = SelectFromModel(LogisticRegression(C=1.0, penalty="l1", solver="liblinear"), value)
                embeded_selector.fit(self.x_train, self.y_train)

                feature_sel_embed = self.x_train.columns[embeded_selector.get_support()]
                re = LogisticRegression().fit(x_train, y_train)
                reg = LogisticRegression().fit(embeded_selector.transform(x_train), y_train)
                print(f"特長量 : {feature_sel_embed}")

                print(f"Lasso回帰による特長選択で{len(self.x_train.columns)}から{len(feature_sel_embed)}に次元削減しました。")
                
                try:

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_train, y_train), reg.score(embeded_selector.transform(x_train), y_train)], 
                                 title="train metrics")
                    fig.show()

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_val, y_val), reg.score(embeded_selector.transform(x_val), y_val)], 
                                 title="validation metrics")
                    fig.show()
                except:
                    print("not modules plotly")
                
                finally:
                    return self.x_train[feature_sel_embed], self.x_val[feature_sel_embed], self.x_test[feature_sel_embed]
            
        else:
        
            x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, random_state=0)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

            if method == "tree":

                embeded_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0, min_samples_leaf=50), "value")
                embeded_selector.fit(self.x_train, self.y_train)

                feature_sel_embed = self.x_train.columns[embeded_selector.get_support()]
                re = LogisticRegression().fit(x_train, y_train)
                reg = LogisticRegression().fit(embeded_selector.transform(x_train), y_train)
                print(f"特長量 : {feature_sel_embed}")

                print(f"決定木による特長選択で{len(self.x_train.columns)}から{len(feature_sel_embed)}に次元削減しました。")
                
                try:

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_train, y_train), reg.score(embeded_selector.transform(x_train), y_train)], 
                                 title="train metrics")
                    fig.show()

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_val, y_val), reg.score(embeded_selector.transform(x_val), y_val)], 
                                 title="validation metrics")
                    fig.show()
                except:
                    print("not modules plotly")
                    
                finally:
                    return self.x_train[feature_sel_embed], self.x_test[feature_sel_embed]

            else:
                embeded_selector = SelectFromModel(LogisticRegression(C=1.0, penalty="l1", solver="liblinear"), value)
                embeded_selector.fit(self.x_train, self.y_train)

                feature_sel_embed = self.x_train.columns[embeded_selector.get_support()]
                re = LogisticRegression().fit(x_train, y_train)
                reg = LogisticRegression().fit(embeded_selector.transform(x_train), y_train)
                print(f"特長量 : {feature_sel_embed}")

                print(f"Lasso回帰による特長選択で{len(self.x_train.columns)}から{len(feature_sel_embed)}に次元削減しました。")
                
                try:

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_train, y_train), reg.score(embeded_selector.transform(x_train), y_train)], 
                                 title="train metrics")
                    fig.show()

                    fig = px.bar(x=[f"feature={len(self.x_train.columns)}", f"feature={len(feature_sel_embed)}"],
                                 y=[re.score(x_val, y_val), reg.score(embeded_selector.transform(x_val), y_val)], 
                                 title="validation metrics")
                    fig.show()
                except:
                    print("not modules plotly")
                finally:
                    return self.x_train[feature_sel_embed], self.x_test[feature_sel_embed]

            