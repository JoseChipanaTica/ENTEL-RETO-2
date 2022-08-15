import pandas as pd
import numpy as np

v1 = pd.read_csv('final-v1.csv')
v2 = pd.read_csv('entel_v1_1_2.csv')
v4 = pd.read_csv('entel_v1_mlp_2.csv')
v5 = pd.read_csv('entel_last.csv')

v1['Demanda'] = np.round(v1['Demanda'] * 0.20 + \
                         v2['Demanda'] * 0.30 + \
                         v4['Demanda'] * 0.30 + \
                         v5['Demanda'] * 0.2)

v1.to_csv('final.csv', index=False)
