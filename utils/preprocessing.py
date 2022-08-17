from sklearn.preprocessing import RobustScaler


def data_sequence_to_models(x_train, x_test, n):
    """
    Remove las variables correlacionadas y estadariza para una mejor precisión con los modelos de Deep Learning

    :param x_train:
    :param x_test:
    :param n: si n es mayor a 0 entonces las variables anteriores a n no contaran para ver la correlación y ser removidas
    :return: x_train, x_test con las variables importantes
    """
    correlated_features = set()
    correlation_matrix = x_train.iloc[:, n:].corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.95:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    x_train_model = x_train.drop(labels=correlated_features, axis=1)
    x_test_model = x_test.drop(labels=correlated_features, axis=1)

    print(f'TRAIN SHAPE: {x_train_model.shape}')

    sc = RobustScaler()
    _x_train = sc.fit_transform(x_train_model)
    _x_test = sc.transform(x_test_model)

    return _x_train, _x_test
