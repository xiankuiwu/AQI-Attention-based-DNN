import random,os

import optuna
import numpy as np
import pandas as pd

import tensorflow as tf
import torch
from NBEATS import NeuralBeats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def set_seed():
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



class Trainer():
    step_size = 7
    in_dim = 7

    def __init__(self, args=None):
        self.args = args
        self.load_data()

    def load_data(self):
        data = pd.read_csv('BJ.csv').iloc[:, [1,3,4,5,6,7,8]]
        self.labels = data['AQI'].values.reshape(-1,1)

        # data = data.values
        # self.ss = StandardScaler()
        # self.data = self.ss.fit_transform(data)
        # ts_x, ts_y = self.create_dataset(self.data, self.step_size)
        # n = round(len(ts_x) * 0.8)
        # self.ts_x_train, self.ts_x_valid = ts_x[:n], ts_x[n:]
        # self.ts_y_train, self.ts_y_valid = ts_y[:n], ts_y[n:]

        n = round(len(self.labels) * 0.8)
        self.train_data = self.labels[:n]
        self.valid_data = self.labels[n:]


    @staticmethod
    def create_dataset(data, step_size):
        res_x, res_y = [], []
        for i in range(len(data) - step_size):
            res_x.append(data[i:i + step_size])
            res_y.append(data[i + step_size, 0])
        return np.array(res_x), np.array(res_y)

    def objective(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        thetas_dim1 = trial.suggest_int('thetas_dim1', 2, 8)
        thetas_dim2 = trial.suggest_int('thetas_dim2', 2, 8)
        hidden_layer_units = trial.suggest_int('hidden_layer_units', 32, 256,32)
        nb_blocks_per_stack = trial.suggest_int('nb_blocks_per_stack',1,5)
        batch_size = trial.suggest_int('batch_size', 64, 256,32)

        model = NeuralBeats(
            data=self.train_data,
            nb_blocks_per_stack=nb_blocks_per_stack,
            thetas_dims=[thetas_dim1, thetas_dim2],
            hidden_layer_units=hidden_layer_units,
            batch_size=batch_size,

            backcast_length=48,
            forecast_length=1,
            mode='cuda',
        )
        model.fit(
            epoch=10,
            optimiser=torch.optim.Adam(model.net.parameters(),lr=lr)
        )

        x = []
        y = []
        for i in range(len(self.valid_data)-48):
            x.append(self.valid_data[i:i+48])
            y.append(self.valid_data[i+48])
        x = np.squeeze(np.array(x),axis=-1)
        y = np.squeeze(np.array(y),axis=-1)

        _,predict = model.net(torch.from_numpy(x).float() / model.norm_constant)
        predict = predict.detach().cpu().numpy()
        mse = mean_squared_error(y,model.norm_constant * predict)
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
T.history.trials_dataframe().to_csv('NBEATS.csv',index=False,index_label=False)