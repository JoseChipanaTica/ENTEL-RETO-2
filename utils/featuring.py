import numpy as np
import pandas as pd


def featuring(df_train, df, x_base):
    df = pd.concat([df_train.iloc[:, :5], df], axis=1)

    df_z_punto_venta = df.groupby(['Z_PUNTO_VENTA'])[df.iloc[:, 5:].columns].transform('max')
    df_z_modelo = df.groupby(['Z_MODELO'])[df.iloc[:, 5:].columns].transform('max')
    df_z_gama = df.groupby(['Z_GAMA'])[df.iloc[:, 5:].columns].transform('max')
    df_z_marca = df.groupby(['Z_MARCA'])[df.iloc[:, 5:].columns].transform('max')
    df_z_departamento = df.groupby(['Z_DEPARTAMENTO'])[df.iloc[:, 5:].columns].transform('max')

    df_z_s_punto_venta = df.groupby(['Z_PUNTO_VENTA'])[df.iloc[:, 5:].columns].transform('sum')
    df_z_s_modelo = df.groupby(['Z_MODELO'])[df.iloc[:, 5:].columns].transform('sum')
    df_z_s_gama = df.groupby(['Z_GAMA'])[df.iloc[:, 5:].columns].transform('sum')
    df_z_s_marca = df.groupby(['Z_MARCA'])[df.iloc[:, 5:].columns].transform('sum')
    df_z_s_departamento = df.groupby(['Z_DEPARTAMENTO'])[df.iloc[:, 5:].columns].transform('sum')

    df_b_punto_venta = df['Z_PUNTO_VENTA'].apply(lambda x: 1 if x in
                                                                [
                                                                    'da45328ba820604eb99694768f2a430cd933d161601dcb8491b4a9b555232c59',
                                                                    'e1f2d2708f545ddc1d7266ba0cc5ccc88147b77fdf3450e68a974e93018ecf60'] else 0)
    df_b_departameto = df['Z_DEPARTAMENTO'].apply(lambda x: 1 if x in
                                                                 [
                                                                     'd6c21b948958417ca98b682a573eb8aa1084b292d32f760f253ef53da13e5589'] else 0)

    Z_MARCA = df['Z_MARCA'].replace(df['Z_MARCA'].value_counts(normalize=True).to_dict())
    Z_GAMA = df['Z_GAMA'].replace(df['Z_GAMA'].value_counts(normalize=True).to_dict())
    Z_MODELO = df['Z_MODELO'].replace(df['Z_MODELO'].value_counts(normalize=True).to_dict())
    Z_DEPARTAMENTO = df['Z_DEPARTAMENTO'].replace(df['Z_DEPARTAMENTO'].value_counts(normalize=True).to_dict())
    Z_PUNTO_VENTA = df['Z_PUNTO_VENTA'].replace(df['Z_PUNTO_VENTA'].value_counts(normalize=True).to_dict())

    df_max = df.iloc[:, 5:].max(axis=1)
    df_sum = df.iloc[:, 5:].sum(axis=1)
    df_std = df.iloc[:, 5:].std(axis=1)
    df_mean = df.iloc[:, 5:].mean(axis=1)

    df_total = df_sum.apply(lambda x: 1 if x > 0 else 0)
    df_count = df.iloc[:, 5:].stack().apply(lambda x: x if x > 0 else np.nan).unstack(level=1).count(axis=1)

    features = df.iloc[:, 5:].stack().apply(lambda x: x if x < x_base else x_base).unstack(level=1)

    df_z = pd.concat([
        features,
        df_z_punto_venta,
        df_z_modelo,
        df_z_gama,
        df_z_marca,
        df_z_departamento,

        df_z_s_punto_venta,
        df_z_s_modelo,
        df_z_s_gama,
        df_z_s_marca,
        df_z_s_departamento,

        df_b_punto_venta,
        df_b_departameto,

        Z_MARCA,
        Z_GAMA,
        Z_MODELO,
        Z_DEPARTAMENTO,
        Z_PUNTO_VENTA,

        df_max,
        df_sum,
        df_std,
        df_mean,

        df_total,
        df_count

    ], axis=1).T.reset_index(drop=True).T

    return df_z
