import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras as keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Input
import random, os

import optuna


def set_seed():
    random_seed = 10
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_model(
        X,
        hidden_unit1=128,
        hidden_unit2=128,
        dropout=0.1,
):
    x_input = Input(shape=(X.shape[1], X.shape[2]))
    x_input1 = mlp(x_input, hidden_units=[hidden_unit1, 128], dropout_rate=0)
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=128, dropout=dropout
    )(x_input1, x_input1)
    x2 = layers.Add()([attention_output, x_input1])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=[hidden_unit2, 128], dropout_rate=0.0001)
    encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    logits = layers.Dense(1)(representation)
    return Model(inputs=x_input, outputs=logits)


class Trainer():
    step_size = 7
    in_dim = 7
    epochs = 20

    def __init__(self):
        self.load_data()

    def load_data(self):
        data = pd.read_csv('data.csv').iloc[:,1:].values
        self.ss = StandardScaler()
        self.data = self.ss.fit_transform(data)
        ts_x, ts_y = self.create_dataset(self.data, self.step_size)
        n = round(len(ts_x) * 0.8)
        self.ts_x_train, self.ts_x_valid = ts_x[:n], ts_x[n:]
        self.ts_y_train, self.ts_y_valid = ts_y[:n], ts_y[n:]

    @staticmethod
    def create_dataset(data, step_size):
        res_x, res_y = [], []
        for i in range(len(data) - step_size):
            res_x.append(data[i:i + step_size])
            res_y.append(data[i + step_size, -1])
        return np.array(res_x), np.array(res_y)
    def objective(self, trial):
        set_seed()
        hidden_unit1 = trial.suggest_int('hidden_unit1', 16, 256, 16)
        hidden_unit2 = trial.suggest_int('hidden_unit2', 16, 256, 16)
        dropout = trial.suggest_uniform('dropout', 0.0, 1.0)
        batch_size = trial.suggest_int('batch_size', 16, 256, 16)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        model = transformer_model(self.ts_x_train, hidden_unit1, hidden_unit2, dropout)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        model.fit(
            self.ts_x_train, self.ts_y_train,
            validation_data=(self.ts_x_valid, self.ts_y_valid),
            batch_size=batch_size,
            epochs=self.epochs
        )
        predict = model.predict(self.ts_x_valid)
        mse = mean_squared_error(self.ts_y_valid, predict)

        return mse

    def optimizer_optuna(self):
        algo = optuna.samplers.TPESampler()
        study = optuna.create_study(
            sampler=algo,direction='minimize'
        )

        study.optimize(
            self.objective,
            n_trials=500,
            show_progress_bar=True
        )
        self.history = study


T = Trainer()
T.optimizer_optuna()

