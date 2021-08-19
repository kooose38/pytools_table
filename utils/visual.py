import pandas as pd 
import plotly.express as px 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")

class Visual:
    def __init__(self, df, name, plot_type=None):
        """
        最初の生データの可視化を行うことで分布の確認をする
        """
        self.name = name 
        self.plot_type = plot_type
        
        if self.name == "category":
            
            columns = df.select_dtypes(["object", "bool", "category"]).columns.to_list()
            self.df = df[columns].dropna()
            self.plot_category(columns)
            
        elif self.name == "float":
            
            columns = df.select_dtypes(["float64"]).columns.to_list()
            self.df = df[columns].dropna()
            self.plot_float(columns)
            
        elif self.name == "int":
            
            columns = df.select_dtypes(["int64"]).columns.to_list()
            self.df = df[columns].dropna()
            self.plot_int(columns)
            
        elif self.name == "describe":
            
            self.df = df 
            print("統計量:")
            print(self.df.describe(include="all"))
            print("-"*40)
            print("欠損値:")
            print(self.df.isnull().sum() / self.df.shape[0])
            
        elif self.name == "box":
            self.df = df
            self.plot_box()
        else:
            print("第二引数にcategory, float, int, describe, boxのいずれかを入力してください。")
            
        
        
    def plot_category(self, columns):
        
        try:
            
            for col in columns:
                col_df = self.df[[col]].value_counts()
                fig = px.pie(values=col_df.values,
                            labels=col_df.index,
                             title=col + ":", 
                             names=list(col_df.index))
                fig.show()

                print(f"Missing values: {self.df[col].isna().sum()}")
                print(f"Unique values : {len(self.df[col].unique())}")
        except:
            print("該当するデータがありません")
            
    def plot_int(self, columns):
        
        try:
            if self.plot_type == "hist":

                for col in columns:
                    plt.figure()
                    plt.title(col)
                    sns.distplot(self.df[col])
                    
            elif self.plot_type == "box":
                for col in columns:
                    plt.figure()
                    plt.title(col)
                    sns.boxplot(self.df[col])
            else:
                print("第三引数にhist, boxのいずれかを入力してください。")
        except:
            print("該当するデータがありません")
            
    def plot_float(self, columns):
        
        try:
            if self.plot_type == "hist":

                for col in columns:
                    plt.figure()
                    plt.title(col)
                    sns.distplot(self.df[col])
                    
            elif self.plot_type == "box":
                for col in columns:
                    plt.figure()
                    plt.title(col)
                    sns.boxplot(self.df[col])
            else:
                print("第三引数にhist, boxのいずれかを入力してください。")
        except:
            print("該当するデータがありません")
    
    def plot_box(self):
        
        plt.figure(figsize=(10, 10))
        self.df.boxplot()
        plt.xticks(rotation=70)
        plt.show()
        
        