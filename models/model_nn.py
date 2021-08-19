#!pip install keras 
import pandas as pd 
import numpy as np 
import os 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from keras.layers.advanced_activations import ReLU, PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import keras
import optuna 

class KerasNet:
    def __init__(self):
        """
        train(): kerasによる学習とモデルの保存。パラメータを指定して学習し、テストデータに提出.csvの作成を行う。
        optuna(): optunaの評価指標Lossでハイパーパラメータ探索を行う。
        load_model(): 保存したモデルの読み込みをします。
        
        validation_dateが指定ならば引数に入力してください。なければホールド分割します。
        回帰、二値、多値には自動で対応します。
        """
        self.model_path = "" 
    
    def forward(self, param, x_train, y_train):
        model = Sequential()
        #input_layer
        model.add(Dropout(param["input_dropout"], input_shape=(x_train.shape[1], )))
        for i in range(param["hidden_layers"]):
            model.add(Dense(param["hidden_units"]))
            if param["batch_norm"] == "before_act":
                model.add(BatchNormalization())
            if param["hidden_activation"] == "prelu":
                model.add(PReLU())
            elif param["hidden_activation"] == "relu":
                model.add(ReLU())
            else:
                raise NotImplementedError
            model.add(Dropout(param["hidden_dropout"]))
        #output_layer   
        if len(y_train.value_counts().index) <= 2:
            model.add(Dense(1, activation="sigmoid"))
        elif len(y_train.value_counts().index) <= 10:
            model.add(Dense(units=len(y_train.value_counts().index)))
            model.add(Activation("softmax"))
        else:
            model.add(Dense(1))
        return model
                       
            
    def train(self,
              x_train,
              y_train,
              x_val=pd.DataFrame(), 
              y_val=pd.DataFrame(),
              x_test=pd.DataFrame(),
             param={}):
           
        col = x_train.columns
        
        if (x_val.shape[0] > 0) or (x_val.shape[0] > 0):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col)
            x_val = scaler.transform(x_val)
            x_val = pd.DataFrame(x_val, columns=col)

            if len(y_train.value_counts().index) <= 2:
                yy_train = y_train.copy()
            elif len(y_train.value_counts().index) <= 10: #多値クラス分類はOnehotに変換する
                yy_train = y_train.copy()
                yy_train = keras.utils.to_categorical(yy_train, num_classes=len(y_train.value_counts().index))
                y_val = keras.utils.to_categorical(y_val, num_classes=len(y_train.value_counts().index))
            else:
                yy_train = y_train.copy()
            if x_test.shape[0] > 0:
                x_test = scaler.transform(x_test)
                x_test = pd.DataFrame(x_test, columns=col)
                
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col)
            random_state = np.random.randint(1, 59, 1)[0]
            if len(y_train.value_counts().index) <= 2:
                yy_train = y_train.copy()
            elif len(y_train.value_counts().index) <= 10:
                yy_train = y_train.copy()
                yy_train = keras.utils.to_categorical(yy_train, num_classes=len(y_train.value_counts().index))
            else:
                yy_train = y_train.copy()
            x_train, x_val, yy_train, y_val = train_test_split(x_train, yy_train, random_state=random_state)
            if x_test.shape[0] > 0:
                x_test = scaler.transform(x_test)
                x_test = pd.DataFrame(x_test, columns=col)
                
        if len(param) == 0:
            #デフォルトのパラメータ指定値
            param = {
                "input_dropout": 0.0,
                "hidden_layers": 3,
                "hidden_units": 96,
                "hidden_activation": "relu",
                "hidden_dropout": .2,
                "batch_norm": "before_act",
                "optimizer": {"type": "adam", "lr": .001},
                "batch_size": 32
            }
        
        model = self.forward(param, x_train, y_train)
        print(model.summary())
            
        if param["optimizer"]["type"] == "sgd":
            optimizer = SGD(lr=param["optimizer"]["lr"], decay=1e-5, momentum=0.8, nesterov=True)
        elif param["optimizer"]["type"] == "adam":
            optimizer = Adam(lr=param["optimizer"]["lr"], beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError
            
          
        if len(y_train.value_counts().index) <= 2:
            model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        elif len(y_train.value_counts().index) <= 10:
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        else:
            model.compile(loss="mse", optimizer=optimizer)
            
        n_epoch = 200
        patience = 20
        early_stopping = EarlyStopping(patience=patience,
                                       restore_best_weights=True,
                                       monitor="val_loss",
                                       mode="auto")
        
        history = model.fit(x_train,
                  yy_train,
                  epochs=n_epoch,
                  batch_size=param["batch_size"], verbose=1,
                  validation_data=(x_val, y_val),
                 callbacks=[early_stopping])
        
        if x_test.shape[0] > 0:
            pred = model.predict(x_test, batch_size=param["batch_size"])
            if len(y_train.value_counts().index) <= 2:
                pred = np.array(pred)
                pred = np.where(pred >= .5, 1, 0)
            elif len(y_train.value_counts().index) <= 10:
                pred = model.predict_classes(x_test, batch_size=param["batch_size"])
                pred = np.array(pred)
            else:
                pred = np.array(pred)
                
            x_test["predict_nn"] = pred 
            x_test.to_csv("./submission/x_test_submission_nn.csv", index=False)
            
        
        pred = model.predict(x_val)
        loss, acc = model.evaluate(x_val, y_val)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))

        self.plot_metrics(history)
        self.save_model(model)
        
    def save_model(self, model):
        print("モデルの保存を行います。")
        os.makedirs("./model", exist_ok=True)
        model.save_weights('./model/nn.hdf5')
        self.model_path = "./model/nn.hdf2"
        model.save(self.model_path)
            
        
    def load_model(self, filename: str):
        model = keras.models.load_model(filename)
        model.summary()
        return model
            
    def plot_metrics(self, history):
        import matplotlib.pyplot as plt
        
        history_dict = history.history

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        
        plt.figure(figsize=(17, 7))
        plt.subplot(1, 2, 1)
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        
    def optuna(self,
               x_train,
               y_train,
               x_val=pd.DataFrame(),
               y_val=pd.DataFrame()):
        
        col = x_train.columns
        
        if (x_val.shape[0] > 0) or (x_val.shape[0] > 0):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col)
            x_val = scaler.transform(x_val)
            x_val = pd.DataFrame(x_val, columns=col)

            if len(y_train.value_counts().index) <= 2:
                yy_train = y_train.copy()
            elif len(y_train.value_counts().index) <= 10: #多値クラス分類はOnehotに変換する
                yy_train = y_train.copy()
                yy_train = keras.utils.to_categorical(yy_train, num_classes=len(y_train.value_counts().index))
                y_val = keras.utils.to_categorical(y_val, num_classes=len(y_train.value_counts().index))
            else:
                yy_train = y_train.copy()
         
                
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col)
            random_state = np.random.randint(1, 59, 1)[0]
            if len(y_train.value_counts().index) <= 2:
                yy_train = y_train.copy()
            elif len(y_train.value_counts().index) <= 10:
                yy_train = y_train.copy()
                yy_train = keras.utils.to_categorical(yy_train, num_classes=len(y_train.value_counts().index))
            else:
                yy_train = y_train.copy()
            x_train, x_val, yy_train, y_val = train_test_split(x_train, yy_train, random_state=random_state)
            
        def objective_variable(x_train, y_train, yy_train, x_val, y_val):
            
            def objective(trial):
                input_dropout = trial.suggest_uniform("input_dropout", 0.0, 0.2)
                hidden_layers = trial.suggest_int("hidden_layers", 2, 5)
                hidden_units = trial.suggest_int("hidden_units", 32, 256)
                hidden_activation = trial.suggest_categorical("hidden_activation", ["prelu", "relu"])
                hidden_dropout = trial.suggest_uniform("hidden_dropout", 0.0, 0.3)
                eta = trial.suggest_uniform("eta", 0.0001, 0.1)
                batch_norm = trial.suggest_categorical("batch_norm", ["before_act", "no"])
                optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])
                batch_size = trial.suggest_int("batch_size", 32, 128)

                param = {
                    "input_dropout": input_dropout,
                    "hidden_layers": hidden_layers,
                    "hidden_units": hidden_units,
                    "hidden_activation": hidden_activation,
                    "hidden_dropout": hidden_dropout,
                    "batch_norm": batch_norm,
                    "optimizer": optimizer,
                    "batch_size": batch_size
                }
                
                model = self.forward(param, x_train, y_train)
#                 print(model.summary())
                
                if param["optimizer"] == "sgd":
                    optimizer = SGD(lr=eta, decay=1e-5, momentum=0.8, nesterov=True)
                elif param["optimizer"] == "adam":
                    optimizer = Adam(lr=eta, beta_1=0.9, beta_2=0.999, decay=0.)
                else:
                    raise NotImplementedError
            
          
                if len(y_train.value_counts().index) <= 2:
                    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
                elif len(y_train.value_counts().index) <= 10:
                    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
                else:
                    model.compile(loss="mse", optimizer=optimizer)
                    
                n_epoch = 200
                patience = 20
                early_stopping = EarlyStopping(patience=patience,
                                       restore_best_weights=True,
                                       monitor="val_loss",
                                       mode="auto")
        
                history = model.fit(x_train,
                          yy_train,
                          epochs=n_epoch,
                          batch_size=param["batch_size"], verbose=1,
                          validation_data=(x_val, y_val),
                         callbacks=[early_stopping])
            
                loss, acc = model.evaluate(x_val, y_val)
                return loss 
                
            return objective
        
        
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective_variable(x_train, y_train, yy_train, x_val, y_val), n_trials=50)
        
        print(f"ロス値: {study.best_value}")
        print(f"パラメータ: {study.best_params}")
#         print(f"trial: {study.best_trial}")
        
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
        
        