import pandas as pd
import numpy as np

from utils.featuring import featuring
from utils.preprocessing import data_sequence_to_models
from utils.training import training

CONFIG_KAGGLE = {
    'TRAIN_PATH': '/kaggle/input/datathon-entel-2022-reto2/train.csv',
    'TEST_PATH': '/kaggle/input/datathon-entel-2022-reto2/test.csv',
    'SAMPLE_SUBMISSION': '/kaggle/input/datathon-entel-2022-reto2/test_sample.csv'
}

CONFIG = {
    'TRAIN_PATH': '../data/train.csv',
    'TEST_PATH': '../data/test.csv',
    'SAMPLE_SUBMISSION': '../data/test_sample.csv'
}

df_train = pd.read_csv(CONFIG['TRAIN_PATH'])
df_test = pd.read_csv(CONFIG['TEST_PATH'])
df_sub = pd.read_csv(CONFIG['SAMPLE_SUBMISSION'])


def model_lstm_based_40():
    """
    Ejecuta el modelo de LSTM con 40 variables temporales

    :return: none
    """
    n = 40
    x_base = 100

    x_train = featuring(df_train, df_train.iloc[:, 5:45], x_base)
    x_test = featuring(df_train, df_train.iloc[:, 15:55], x_base)

    y_train = df_train.iloc[:, 45:55]

    x_train_data, x_test_data = data_sequence_to_models(x_train, x_test, n)

    y_train = y_train.stack().apply(lambda x: x if x < x_base else x_base).unstack(level=1).values

    models = training('lstm', x_train_data, y_train, n)

    lstm_test_csv(x_test_data, models, n, 'lstm_40')


def model_lstm_based_20():
    """
    Ejecuta un modelo LSTM con 20 variables temporales

    :return:
    """
    n = 20
    x_base = 100

    x_train = pd.concat([
        featuring(df_train, df_train.iloc[:, 5:25], x_base),
        featuring(df_train, df_train.iloc[:, 15:35], x_base),
        featuring(df_train, df_train.iloc[:, 25:45], x_base),
    ], axis=0).reset_index(drop=True)

    y_train = pd.concat([
        pd.DataFrame(df_train.iloc[:, 25:35].values),
        pd.DataFrame(df_train.iloc[:, 35:45].values),
        pd.DataFrame(df_train.iloc[:, 45:55].values),
    ], axis=0).reset_index(drop=True)

    x_test = featuring(df_train, df_train.iloc[:, 35:55], x_base)

    x_train_data, x_test_data = data_sequence_to_models(x_train, x_test, n)

    y_train = y_train.stack().apply(lambda x: x if x < x_base else x_base).unstack(level=1).values

    models = training('lstm', x_train_data, y_train, n)

    lstm_test_csv(x_test_data, models, n, 'lstm_20')


def model_lstm_based_10():
    """
    Ejecuta un modelo LSTM con 10 variables temporales
    :return:
    """
    n = 10
    x_base = 100

    x_train = pd.concat([
        featuring(df_train, df_train.iloc[:, 5:15], x_base),
        featuring(df_train, df_train.iloc[:, 15:25], x_base),
        featuring(df_train, df_train.iloc[:, 25:35], x_base),
        featuring(df_train, df_train.iloc[:, 35:45], x_base),
    ], axis=0).reset_index(drop=True)

    y_train = pd.concat([
        pd.DataFrame(df_train.iloc[:, 15:25].values),
        pd.DataFrame(df_train.iloc[:, 25:35].values),
        pd.DataFrame(df_train.iloc[:, 35:45].values),
        pd.DataFrame(df_train.iloc[:, 45:55].values)
    ], axis=0).reset_index(drop=True)

    x_test = featuring(df_train, df_train.iloc[:, 45:55], x_base)

    x_train_data, x_test_data = data_sequence_to_models(x_train, x_test, n)

    y_train = y_train.stack().apply(lambda x: x if x < x_base else x_base).unstack(level=1).values

    models = training('lstm', x_train_data, y_train, n)

    lstm_test_csv(x_test_data, models, n, 'lstm_10')


def model_mlp_based_40():
    """
    Ejecuta un modelo MLP con las 40 variables temporales
    :return:
    """
    n = 0
    x_base = 100

    x_train = pd.concat([
        featuring(df_train, df_train.iloc[:, 5:25], x_base),
        featuring(df_train, df_train.iloc[:, 15:35], x_base),
        featuring(df_train, df_train.iloc[:, 25:45], x_base),
    ], axis=0).reset_index(drop=True)

    y_train = pd.concat([
        pd.DataFrame(df_train.iloc[:, 25:35].values),
        pd.DataFrame(df_train.iloc[:, 35:45].values),
        pd.DataFrame(df_train.iloc[:, 45:55].values),
    ], axis=0).reset_index(drop=True)

    x_test = featuring(df_train, df_train.iloc[:, 35:55], x_base)

    x_train_data, x_test_data = data_sequence_to_models(x_train, x_test, n)

    y_train = y_train.stack().apply(lambda x: x if x < x_base else x_base).unstack(level=1).values

    models = training('mlp', x_train_data, y_train, n)

    mlp_test_csv(x_test_data, models, n, 'mlp_40')


def model_mlp_based_20():
    """
    Ejecuta un modelo MLP con 20 variables temporales
    :return:
    """
    n = 0
    x_base = 100

    x_train = pd.concat([
        featuring(df_train, df_train.iloc[:, 5:15], x_base),
        featuring(df_train, df_train.iloc[:, 15:25], x_base),
        featuring(df_train, df_train.iloc[:, 25:35], x_base),
        featuring(df_train, df_train.iloc[:, 35:45], x_base),
    ], axis=0).reset_index(drop=True)

    y_train = pd.concat([
        pd.DataFrame(df_train.iloc[:, 15:25].values),
        pd.DataFrame(df_train.iloc[:, 25:35].values),
        pd.DataFrame(df_train.iloc[:, 35:45].values),
        pd.DataFrame(df_train.iloc[:, 45:55].values)
    ], axis=0).reset_index(drop=True)

    x_test = featuring(df_train, df_train.iloc[:, 45:55], x_base)

    x_train_data, x_test_data = data_sequence_to_models(x_train, x_test, n)

    y_train = y_train.stack().apply(lambda x: x if x < x_base else x_base).unstack(level=1).values

    models = training('mlp', x_train_data, y_train, n)

    mlp_test_csv(x_test_data, models, n, 'mlp_20')


def model_mlp_based_10():
    """
    Ejecuta un modelo MLP con 10 variables temporales
    :return:
    """
    n = 0
    x_base = 100

    x_train = featuring(df_train, df_train.iloc[:, 5:45], x_base)
    x_test = featuring(df_train, df_train.iloc[:, 15:55], x_base)

    y_train = df_train.iloc[:, 45:55]

    x_train_data, x_test_data = data_sequence_to_models(x_train, x_test, n)

    y_train = y_train.stack().apply(lambda x: x if x < x_base else x_base).unstack(level=1).values

    models = training('mlp', x_train_data, y_train, n)

    mlp_test_csv(x_test_data, models, n, 'mlp_10')


def lstm_test_csv(x_test, models, n, name):
    """
    Ejecuta los datos test en los modelos creados

    :param x_test:
    :param models:
    :param n:
    :param name:
    :return:
    """
    predictions = []

    x_test_features = x_test[:, :n]
    x_test_features = np.reshape(x_test_features, (x_test_features.shape[0], x_test_features.shape[1], 1))

    x_test_extras = x_test[:, n:]

    for model in models:
        _pred = model.predict([x_test_features, x_test_extras])
        predictions.append(_pred)

    sub_predictions = (predictions[0] + predictions[1] + predictions[2] + predictions[3] + predictions[4]) / 5

    result_to_csv(sub_predictions, name)


def mlp_test_csv(x_test, models, n, name):
    """
    Ejecuta los datos test en los modelos creados

    :param x_test:
    :param models:
    :param n:
    :param name:
    :return:
    """
    predictions = []
    for model in models:
        _pred = model.predict([x_test])
        predictions.append(_pred)

    sub_predictions = (predictions[0] + predictions[1] + predictions[2] + predictions[3] + predictions[4]) / 5

    result_to_csv(sub_predictions, name)


def result_to_csv(predictions, name):
    """
    Crea un csv con los resultados

    :param predictions: array
    :param name: str
    :return:
    """
    df_submission = pd.merge(df_train.iloc[:, :5],
                             pd.DataFrame(predictions),
                             how='inner',
                             left_index=True,
                             right_index=True)

    df_submission = df_submission.rename(columns={
        0: 'SEMANA_51',
        1: 'SEMANA_52',
        2: 'SEMANA_53',
        3: 'SEMANA_54',
        4: 'SEMANA_55',
        5: 'SEMANA_56',
        6: 'SEMANA_57',
        7: 'SEMANA_58',
        8: 'SEMANA_59',
        9: 'SEMANA_60'
    })

    df_submission['BASE_ID'] = df_submission['Z_MODELO'].astype(str) + '|' + \
                               df_submission['Z_PUNTO_VENTA'].astype(str) + '|' + \
                               df_submission['Z_GAMA'].astype(str)

    df_submission = df_submission.iloc[:, 5:]

    df_submission = df_submission.set_index('BASE_ID').stack().to_frame().reset_index()
    df_submission['BASE_ID'] = df_submission['BASE_ID'].astype(str) + '|' + df_submission['level_1'].astype(str)

    df_submission = df_submission.drop(['level_1'], axis=1)
    df_submission.columns = ['ID', 'Demanda']

    df_submission.to_csv(f'{name}.csv', index=False)


if __name__ == '__main__':
    print('Start utils')

    model_lstm_based_40()
    model_lstm_based_20()
    model_lstm_based_10()
    model_mlp_based_40()
    model_mlp_based_20()
    model_mlp_based_10()
