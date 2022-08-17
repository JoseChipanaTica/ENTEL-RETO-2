import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from models import mlp, lstm
from utils import eval_model

EPOCH = 1000
BATCH_SIZE = 512

es = keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                   min_delta=1e-05,
                                   patience=30,
                                   verbose=1,
                                   mode='min',
                                   restore_best_weights=True)
plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error',
                                            factor=0.1,
                                            patience=10,
                                            verbose=1,
                                            min_lr=5e-7,
                                            mode='min')


def data(x_train_data, y_train, t_idx, v_idx, model_type, n=0):
    if model_type == 'mlp':
        _x_train, _x_valid = x_train_data[t_idx], x_train_data[v_idx]

        _y_train, _y_valid = y_train[t_idx], y_train[v_idx]

        model = mlp(_x_train.shape[-1])

        return [_x_train], [_x_valid], _y_train, _y_valid, model

    if model_type == 'lstm':
        _x_train, _x_valid = x_train_data[t_idx], x_train_data[v_idx]

        x_t_features, x_v_features = _x_train[:, :n], _x_valid[:, :n]
        x_t_features = np.reshape(x_t_features, (x_t_features.shape[0], x_t_features.shape[1], 1))
        x_v_features = np.reshape(x_v_features, (x_v_features.shape[0], x_v_features.shape[1], 1))

        x_t_extras, x_v_extras = _x_train[:, n:], _x_valid[:, n:]

        _y_train, _y_valid = y_train[t_idx], y_train[v_idx]

        l_fet = x_t_features.shape[-2:]
        l_ext = x_t_extras.shape[-1]

        model = lstm(l_fet, l_ext)

        return [x_t_features, x_t_extras], [x_v_features, x_v_extras], _y_train, _y_valid, model


def training(model_type, x_train_data, y_train, n):
    """
    :param n: n define the number of column to use for lstm or cnn
    :param y_train: numpy array
    :param x_train_data: numpy array
    :param model_type: str type: could be lstm, cnn or mlp
    :return: list of tensorflow models
    """

    models = []

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

    y_group = pd.Series(y_train.sum(axis=1)).apply(lambda x: x if x < 15 else 15).values

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_data, y_group)):
        print('-' * 15, '>', f'Fold {fold + 1}', '<', '-' * 15)

        _x_train, _x_valid, _y_train, _y_valid, model = data(x_train_data, y_train, train_idx, val_idx, model_type, n)

        model.fit(_x_train, _y_train,
                  validation_data=(_x_valid, _y_valid),
                  epochs=EPOCH,
                  batch_size=BATCH_SIZE,
                  callbacks=[es, plateau],
                  verbose=1)

        eval_model(model, _x_valid, _y_valid)

        models.append(model)

    return models
