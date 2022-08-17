from tensorflow import keras


def lstm(l_features, l_extras_features):
    features = keras.layers.Input(shape=l_features)
    tabular = keras.layers.Input(shape=l_extras_features)

    out_features = keras.layers.LSTM(250, return_sequences=True)(features)
    out_features = keras.layers.Dropout(0.2)(out_features)
    out_features = keras.layers.LSTM(150, return_sequences=True)(out_features)
    out_features = keras.layers.Dropout(0.2)(out_features)
    out_features = keras.layers.LSTM(100)(out_features)
    out_features = keras.layers.Flatten()(out_features)

    out_features = keras.layers.Dense(50, activation='linear')(out_features)
    out_features = keras.layers.Dropout(0.2)(out_features)
    out_features = keras.layers.Dense(32, activation='linear')(out_features)

    for n_hidden in [512, 256, 128, 64, 32]:
        out_tabular = keras.layers.Dense(n_hidden, activation='relu')(tabular)
        out_tabular = keras.layers.BatchNormalization()(out_tabular)
        out_tabular = keras.layers.Dropout(0.2)(out_tabular)

    out = keras.layers.Multiply()([out_features, out_tabular])
    out = keras.layers.Dense(10, activation='relu')(out)

    model = keras.Model(inputs=[features, tabular], outputs=out)

    mse = keras.losses.MeanSquaredError()
    rmse = keras.metrics.RootMeanSquaredError()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0004), loss=mse, metrics=[rmse])

    return model


def mlp(l_extras_features):
    tabular = keras.layers.Input(shape=l_extras_features)

    for n_hidden in [1024, 512, 256, 128, 64, 32]:
        out_tabular = keras.layers.Dense(n_hidden, activation='linear')(tabular)
        out_tabular = keras.layers.BatchNormalization()(out_tabular)
        out_tabular = keras.layers.Dropout(0.2)(out_tabular)

    out = keras.layers.Dense(10, activation='relu')(out_tabular)

    model = keras.Model(inputs=[tabular], outputs=out)

    mse = keras.losses.MeanSquaredError()
    rmse = keras.metrics.RootMeanSquaredError()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0004), loss=mse, metrics=[rmse])

    return model
