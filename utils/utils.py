import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error


def eval_model(model, x_valid, y_valid):
    preds = pd.DataFrame(np.round(model.predict(x_valid)).astype('int32')).stack().reset_index(drop=True)
    y_valid = pd.DataFrame(y_valid).stack().reset_index(drop=True)
    print(f' RMSE --> {mean_squared_error(y_valid, preds, squared=False)}')
