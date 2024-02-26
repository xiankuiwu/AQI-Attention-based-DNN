import random,os

import optuna
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as keras
import torch
from NBEATS import NeuralBeats

from sklearn.metrics import mean_squared_error
from keras import layers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from keras.models import Sequential, Model
from keras.layers import LSTM, Activation, Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Reshape, Concatenate, concatenate

def set_seed():
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_model(x, hidden_1, hidden_2, dropout_rate=0.1, attention_dim=11, num_heads=4):
    x_input = Input(shape=(x.shape[1], x.shape[2]))
    AQI, P = x_input[:, :, 0], x_input[:, :, 1:]
    feature_time_lag = mlp(AQI, [hidden_1, hidden_2], dropout_rate)
    feature_pol_con = mlp(P, [hidden_1, hidden_2], dropout_rate)
    feature_seasonal = mlp(x_input, [hidden_1, hidden_2], dropout_rate)
    x_input1 = layers.Dense(hidden_2)(x_input)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=attention_dim, dropout=0.0799
    )(x_input1, x_input1)
    x2 = layers.Add()([attention_output, feature_time_lag, feature_pol_con, feature_seasonal])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=[hidden_2], dropout_rate=dropout_rate)
    encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    logits = layers.Dense(1)(representation)
    logits = layers.Dropout(dropout_rate)(logits)
    return Model(inputs=x_input, outputs=logits)


class Trainer():
    step_size = 7
    in_dim = 7

    def __init__(self, args=None):
        self.args = args
        self.load_data()

    def load_data(self):
        data = pd.read_csv('BJ.csv').iloc[:, [1,3,4,5,6,7,8]]
        self.labels = data['AQI'].values.reshape(-1,1)

        data = data.values
        self.ss = StandardScaler()
        self.data = self.ss.fit_transform(data)
        ts_x, ts_y = self.create_dataset(self.data, self.step_size)
        n = round(len(ts_x) * 0.8)
        self.ts_x_train, self.ts_x_valid = ts_x[:n], ts_x[n:]
        self.ts_y_train, self.ts_y_valid = ts_y[:n], ts_y[n:]

        # n = round(len(self.labels) * 0.8)
        # self.train_data = self.labels[:n]
        # self.valid_data = self.labels[n:]


    @staticmethod
    def create_dataset(data, step_size):
        res_x, res_y = [], []
        for i in range(len(data) - step_size):
            res_x.append(data[i:i + step_size])
            res_y.append(data[i + step_size, 0])
        return np.array(res_x), np.array(res_y)

    def objective(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        hidden_1 = trial.suggest_int('hidden_1', 64, 256,32)
        hidden_2 = trial.suggest_int('hidden_2', 64, 256,32)
        dropout_rate = trial.suggest_uniform('dropout',0.0,0.8)
        attention_dim = trial.suggest_int('attention_dim',32,256,32)
        num_heads = trial.suggest_int('num_heads',2,8,2)
        batch_size = trial.suggest_int('batch_size', 64, 256,32)

        model = transformer_model(self.ts_x_train,hidden_1, hidden_2, dropout_rate, attention_dim, num_heads)
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=lr))
        model.fit(
            self.ts_x_train,self.ts_y_train,
            validation_data=(self.ts_x_valid, self.ts_y_valid),
            batch_size=batch_size,
            epochs=20,
            verbose=0
        )

        predict = model.predict(self.ts_x_valid)
        mse = mean_squared_error(self.ts_y_valid, predict)
        return mse

    def optimizer_optuna(self):
        algo = optuna.samplers.TPESampler()
        study = optuna.create_study(
            sampler=algo, direction='minimize'
        )
        study.optimize(
            self.objective,
            n_trials=200,
            show_progress_bar=True
        )
        self.history = study


T = Trainer()
T.optimizer_optuna()
T.history.trials_dataframe().to_csv('ADNNet.csv',index=False,index_label=False)