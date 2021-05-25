from tensorflow import keras
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
# from pyunpack import Archive
# Archive('Dataset.rar').extractall('./')
from pathlib import Path
import string
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

cols = ['id', 'date_start', 'date_end', 'day_diff', 'gaz_amount', 'gaz_mean', 'gaz_median', 'gaz_min', 'gaz_max', 'gaz_var', 'gaz_std']
major_cols = ['liquid_tm', 'liquid', 'energy', 'water', 'nd', 'pressure_zad', 'pressure_buf', 'd', 'lin', 'kvch', 'сurrent', 'freq', 'vol']
minor_cols = ['_amount', '_mean', '_median', '_min', '_max', '_var', '_std']
add_cols = []
for major_col in major_cols:
    for minor_col in minor_cols:
        add_cols.append(major_col+minor_col)
cols += add_cols
cols += ['prim', 'target', 'ungerm', 'comment']
data = pd.read_excel('data_otkaz-.xls', skiprows=1, names=cols)

feature_cols = ['day_diff']
major_cols = ['gaz', 'liquid_tm', 'liquid', 'energy', 'water', 'nd', 'pressure_zad', 'pressure_buf', 'd', 'lin', 'kvch', 'сurrent', 'freq', 'vol']
minor_cols = ['_mean', '_median', '_min', '_max', '_var', '_std']
add_cols = []
for major_col in major_cols:
    for minor_col in minor_cols:
        add_cols.append(major_col+minor_col)
feature_cols += add_cols
all_cols = feature_cols + ['target']
filtered_data = data[all_cols]
filled_data = filtered_data.replace('-', 0)

# arr = [day_amount] + ([mean, median, min, max, var, std] for each sensor)
def raw_predict(arr, scaler, model):
    """
    Get probability from raw data and model
    """
    arr = arr.reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    predict = model.predict(arr_scaled)
    return predict[0][0]

def clear_data(df, id):
    """
    Process raw xlsx documents
    """
    df = df.iloc[:, 1:df.shape[1]-1]
    # remove unneeded columns from raw data
    qn = df[11]
    df = df.loc[:, ~df.columns.isin([1, 4, 5, 7, 9, 11, 18])]
    df['qn'] = qn
    df.columns = major_cols
    df['id'] = id
    return df

# Clear all xlsx documents and reformat them into comfortable format
path = Path('./Dataset')
data = []
for pump in os.listdir(path):
    pump_name = pump[:-4]
    for char in pump_name:
        if char not in string.digits:
            break
    if char not in string.digits:
        continue
    raw_data = pd.read_excel(path / pump, sheet_name=2, skiprows=2, header=None)
    data.append(clear_data(raw_data, pump_name))

full_data = pd.concat(data, axis=0).reset_index(drop=True)
full_data.to_csv('FULLDATASET.csv', index=None)

# input format ['gaz', 'liquid_tm', 'liquid', 'energy', 'water', 'nd', 'pressure_zad', 'pressure_buf', 'd', 'lin', 'kvch', 'сurrent', 'freq', 'vol', 'id']
def add_row(full_data, arr):
    """
    Params:
    full_data: pd.DataFrame - raw records data from each pump
    arr: new record to add
    Return:
    full_data: pd.DataFrame - data with new record
    data_for_model: np.array - aggregations for each sensor
    """
    full_data.loc[full_data.shape[0]] = arr
    full_data.to_csv('FULLDATASETADDED.csv', index=None)
    id = arr[-1]
    filtered_data = full_data[full_data['id'] == id]
    filtered_data_gr = filtered_data.groupby(['id'])
    amount = filtered_data['gaz'].count()
    aggregation_cols = ['mean', 'median', 'min', 'max', 'var', 'std']
    agg_dict = {major_col:aggregation_cols for major_col in major_cols}
    agg_values = list(filtered_data_gr.agg(agg_dict).values[0])
    for i in range(4, 84, 6):
        if np.isnan(agg_values[i]):
            agg_values[i] = 0
        else:
            agg_values[i] /= 100
    for i in range(5, 84, 6):
        if np.isnan(agg_values[i]):
            agg_values[i] = 0
    data_for_model = np.array([amount] + agg_values)
    return full_data, data_for_model

def top_n(df, n):
    """
    Count statistics to input into model based on top n % of the data
    Params:
        df : pd.DataFrame with all days for one pump, cols = ['gaz', 'liquid_tm', 'liquid', 'energy', 'water', 'nd', 'pressure_zad', 'pressure_buf', 'd', 'lin', 'kvch', 'сurrent', 'freq', 'vol', 'id']
        n : top n % of data to count statistics for model
    Return:
        data_for_model : np.array, shape = (1, 85) (cols amount)
    """
    shape = df.shape[0]
    top = round(shape * n)
    top_df = df.iloc[:top, :]
    top_next = df.iloc[top, :]
    _, data_for_model = add_row(top_df, top_next)
    return data_for_model.reshape(1, -1)

def between_n(df, n):
    """
    Count statistics to input into model based on top n % of the data
    Params:
        df : pd.DataFrame with all days for one pump, cols = ['gaz', 'liquid_tm', 'liquid', 'energy', 'water', 'nd', 'pressure_zad', 'pressure_buf', 'd', 'lin', 'kvch', 'сurrent', 'freq', 'vol', 'id']
        n : top n % of data to count statistics for model
    Return:
        data_for_model : np.array, shape = (1, 85) (cols amount)
    """
    start = n - 0.15 if n == 0.95 else n - 0.2
    shape = df.shape[0]
    top = round(shape * n)
    bottom = round(shape * start)
    top_df = df.iloc[bottom:top, :]
    top_next = df.iloc[top, :]
    _, data_for_model = add_row(top_df, top_next)
    return data_for_model.reshape(1, -1)

generated_data = []
for id in full_data['id'].unique():
    if id == '766':
        continue
    id_df = full_data[full_data['id'] == id]
    top_20 = between_n(id_df, 0.2)
    top_40 = between_n(id_df, 0.4)
    top_60 = between_n(id_df, 0.6)
    top_80 = between_n(id_df, 0.8)
    top_95 = between_n(id_df, 0.95)
    concat = np.concatenate((top_20, top_40, top_60, top_80, top_95), axis=0)
    df = pd.DataFrame(concat, columns=feature_cols)
    df['target'] = [0, 0, 0, 0, 0] if id == 524 else [0, 0, 0, 0, 1]
    generated_data.append(df)

total_generated_data = pd.concat(generated_data, axis=0)
total_real_and_generated_data = pd.concat([filled_data, total_generated_data], axis=0).reset_index(drop=True)

acc = []
kfold = StratifiedKFold(5, shuffle=True, random_state=174)
for train_idx, test_idx in kfold.split(total_real_and_generated_data[feature_cols],
                                       total_real_and_generated_data['target']):
    X_train = total_real_and_generated_data.loc[train_idx, feature_cols]
    X_test = total_real_and_generated_data.loc[test_idx, feature_cols]

    y_train = total_real_and_generated_data.loc[train_idx, 'target']
    y_test = total_real_and_generated_data.loc[test_idx, 'target']

    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)
    model = keras.Sequential([
        keras.layers.Dense(32, input_shape=(X_train_scaled.shape[1],), activation='relu'),
        keras.layers.Dense(16, input_shape=(32,), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])

    model.fit(
        x=X_train_scaled,
        y=y_train,
        shuffle=False,
        epochs=8,
        batch_size=2
    )
    print('EVALUATION')
    eval = model.evaluate(x=X_test_scaled, y=y_test)
    acc.append(eval[1])

from keras import backend as K

# На всех данных
X = total_real_and_generated_data.loc[:, feature_cols]
y = total_real_and_generated_data.loc[:, 'target']

std_scaler_final = StandardScaler()
X_scaled = std_scaler_final.fit_transform(X)

model_final = keras.Sequential([
        keras.layers.Dense(32, input_shape=(X_scaled.shape[1],), activation='relu'),
        keras.layers.Dense(16, input_shape=(32,), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
])
model_final.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])
K.set_value(model_final.optimizer.learning_rate, 0.01) # значения из диапазона [0.03, 0.001]
model_final.fit(
    x=X_scaled,
    y=y,
    shuffle=True,
    epochs=32,
    batch_size=4,
)

def is_digit(string):
    if string.isdigit():
       return True
    else:
        try:
            float(string)
            return True
        except ValueError:
            return False

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(641, 276)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.pushButton_get_file = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_get_file.setMinimumSize(QtCore.QSize(140, 40))
        self.pushButton_get_file.setObjectName("pushButton_get_file")
        self.pushButton_get_file.clicked.connect(self.on_click_get_file)
        self.horizontalLayout_25.addWidget(self.pushButton_get_file)
        spacerItem = QtWidgets.QSpacerItem(288, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_25.addItem(spacerItem)
        self.pushButton_save_result = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_save_result.setMinimumSize(QtCore.QSize(140, 40))
        self.pushButton_save_result.setObjectName("pushButton_save_result")
        self.pushButton_save_result.clicked.connect(self.on_click_save_result)
        self.horizontalLayout_25.addWidget(self.pushButton_save_result)
        self.verticalLayout_8.addLayout(self.horizontalLayout_25)
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        spacerItem1 = QtWidgets.QSpacerItem(58, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_29.addItem(spacerItem1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_pump_id = QtWidgets.QLabel(self.centralwidget)
        self.label_pump_id.setMinimumSize(QtCore.QSize(50, 0))
        self.label_pump_id.setObjectName("label_pump_id")
        self.horizontalLayout_4.addWidget(self.label_pump_id)
        spacerItem2 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.lineEdit_pump_id = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_pump_id.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_pump_id.setObjectName("lineEdit_pump_id")
        self.horizontalLayout_4.addWidget(self.lineEdit_pump_id)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_Q_gas_TM = QtWidgets.QLabel(self.centralwidget)
        self.label_Q_gas_TM.setMinimumSize(QtCore.QSize(50, 0))
        self.label_Q_gas_TM.setObjectName("label_Q_gas_TM")
        self.horizontalLayout_3.addWidget(self.label_Q_gas_TM)
        spacerItem3 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.lineEdit_Q_gas_TM = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Q_gas_TM.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_Q_gas_TM.setObjectName("lineEdit_Q_gas_TM")
        self.horizontalLayout_3.addWidget(self.lineEdit_Q_gas_TM)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_Q_liq_TM = QtWidgets.QLabel(self.centralwidget)
        self.label_Q_liq_TM.setMinimumSize(QtCore.QSize(50, 0))
        self.label_Q_liq_TM.setObjectName("label_Q_liq_TM")
        self.horizontalLayout_8.addWidget(self.label_Q_liq_TM)
        spacerItem4 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem4)
        self.lineEdit_Q_liq_TM = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Q_liq_TM.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_Q_liq_TM.setObjectName("lineEdit_Q_liq_TM")
        self.horizontalLayout_8.addWidget(self.lineEdit_Q_liq_TM)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_Q_liq = QtWidgets.QLabel(self.centralwidget)
        self.label_Q_liq.setMinimumSize(QtCore.QSize(50, 0))
        self.label_Q_liq.setObjectName("label_Q_liq")
        self.horizontalLayout_5.addWidget(self.label_Q_liq)
        spacerItem5 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem5)
        self.lineEdit_Q_liq = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Q_liq.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_Q_liq.setObjectName("lineEdit_Q_liq")
        self.horizontalLayout_5.addWidget(self.lineEdit_Q_liq)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.label_W_act = QtWidgets.QLabel(self.centralwidget)
        self.label_W_act.setMinimumSize(QtCore.QSize(50, 0))
        self.label_W_act.setObjectName("label_W_act")
        self.horizontalLayout_28.addWidget(self.label_W_act)
        spacerItem6 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_28.addItem(spacerItem6)
        self.lineEdit_W_act = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_W_act.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_W_act.setObjectName("lineEdit_W_act")
        self.horizontalLayout_28.addWidget(self.lineEdit_W_act)
        self.verticalLayout_3.addLayout(self.horizontalLayout_28)
        self.horizontalLayout_29.addLayout(self.verticalLayout_3)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_OBV_HAL = QtWidgets.QLabel(self.centralwidget)
        self.label_OBV_HAL.setMinimumSize(QtCore.QSize(50, 0))
        self.label_OBV_HAL.setObjectName("label_OBV_HAL")
        self.horizontalLayout_6.addWidget(self.label_OBV_HAL)
        spacerItem7 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem7)
        self.lineEdit_OBV_HAL = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_OBV_HAL.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_OBV_HAL.setObjectName("lineEdit_OBV_HAL")
        self.horizontalLayout_6.addWidget(self.lineEdit_OBV_HAL)
        self.verticalLayout_5.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_N_d = QtWidgets.QLabel(self.centralwidget)
        self.label_N_d.setMinimumSize(QtCore.QSize(50, 0))
        self.label_N_d.setObjectName("label_N_d")
        self.horizontalLayout_2.addWidget(self.label_N_d)
        spacerItem8 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem8)
        self.lineEdit_N_d = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_N_d.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_N_d.setObjectName("lineEdit_N_d")
        self.horizontalLayout_2.addWidget(self.lineEdit_N_d)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_P_zatr = QtWidgets.QLabel(self.centralwidget)
        self.label_P_zatr.setMinimumSize(QtCore.QSize(50, 0))
        self.label_P_zatr.setObjectName("label_P_zatr")
        self.horizontalLayout_7.addWidget(self.label_P_zatr)
        spacerItem9 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem9)
        self.lineEdit_P_zatr = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_P_zatr.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_P_zatr.setObjectName("lineEdit_P_zatr")
        self.horizontalLayout_7.addWidget(self.lineEdit_P_zatr)
        self.verticalLayout_5.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_P_buf = QtWidgets.QLabel(self.centralwidget)
        self.label_P_buf.setMinimumSize(QtCore.QSize(50, 0))
        self.label_P_buf.setObjectName("label_P_buf")
        self.horizontalLayout_17.addWidget(self.label_P_buf)
        spacerItem10 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem10)
        self.lineEdit_P_buf = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_P_buf.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_P_buf.setObjectName("lineEdit_P_buf")
        self.horizontalLayout_17.addWidget(self.lineEdit_P_buf)
        self.verticalLayout_5.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_D = QtWidgets.QLabel(self.centralwidget)
        self.label_D.setMinimumSize(QtCore.QSize(50, 0))
        self.label_D.setObjectName("label_D")
        self.horizontalLayout_18.addWidget(self.label_D)
        spacerItem11 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem11)
        self.lineEdit_D = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_D.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_D.setObjectName("lineEdit_D")
        self.horizontalLayout_18.addWidget(self.lineEdit_D)
        self.verticalLayout_5.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_29.addLayout(self.verticalLayout_5)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_P_lin = QtWidgets.QLabel(self.centralwidget)
        self.label_P_lin.setMinimumSize(QtCore.QSize(50, 0))
        self.label_P_lin.setObjectName("label_P_lin")
        self.horizontalLayout_19.addWidget(self.label_P_lin)
        spacerItem12 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_19.addItem(spacerItem12)
        self.lineEdit_P_lin = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_P_lin.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_P_lin.setObjectName("lineEdit_P_lin")
        self.horizontalLayout_19.addWidget(self.lineEdit_P_lin)
        self.verticalLayout_7.addLayout(self.horizontalLayout_19)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.label_KVCH_HAL = QtWidgets.QLabel(self.centralwidget)
        self.label_KVCH_HAL.setMinimumSize(QtCore.QSize(50, 0))
        self.label_KVCH_HAL.setObjectName("label_KVCH_HAL")
        self.horizontalLayout_20.addWidget(self.label_KVCH_HAL)
        spacerItem13 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem13)
        self.lineEdit_KVCH_HAL = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_KVCH_HAL.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_KVCH_HAL.setObjectName("lineEdit_KVCH_HAL")
        self.horizontalLayout_20.addWidget(self.lineEdit_KVCH_HAL)
        self.verticalLayout_7.addLayout(self.horizontalLayout_20)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label_I = QtWidgets.QLabel(self.centralwidget)
        self.label_I.setMinimumSize(QtCore.QSize(50, 0))
        self.label_I.setObjectName("label_I")
        self.horizontalLayout_21.addWidget(self.label_I)
        spacerItem14 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_21.addItem(spacerItem14)
        self.lineEdit_I = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_I.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_I.setObjectName("lineEdit_I")
        self.horizontalLayout_21.addWidget(self.lineEdit_I)
        self.verticalLayout_7.addLayout(self.horizontalLayout_21)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.label_F = QtWidgets.QLabel(self.centralwidget)
        self.label_F.setMinimumSize(QtCore.QSize(50, 0))
        self.label_F.setObjectName("label_F")
        self.horizontalLayout_22.addWidget(self.label_F)
        spacerItem15 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_22.addItem(spacerItem15)
        self.lineEdit_F = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_F.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_F.setObjectName("lineEdit_F")
        self.horizontalLayout_22.addWidget(self.lineEdit_F)
        self.verticalLayout_7.addLayout(self.horizontalLayout_22)
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.label_Q_n = QtWidgets.QLabel(self.centralwidget)
        self.label_Q_n.setMinimumSize(QtCore.QSize(50, 0))
        self.label_Q_n.setObjectName("label_Q_n")
        self.horizontalLayout_23.addWidget(self.label_Q_n)
        spacerItem16 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_23.addItem(spacerItem16)
        self.lineEdit_Q_n = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Q_n.setMaximumSize(QtCore.QSize(70, 16777215))
        self.lineEdit_Q_n.setObjectName("lineEdit_Q_n")
        self.horizontalLayout_23.addWidget(self.lineEdit_Q_n)
        self.verticalLayout_7.addLayout(self.horizontalLayout_23)
        self.horizontalLayout_29.addLayout(self.verticalLayout_7)
        spacerItem17 = QtWidgets.QSpacerItem(58, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_29.addItem(spacerItem17)
        self.verticalLayout_8.addLayout(self.horizontalLayout_29)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        spacerItem18 = QtWidgets.QSpacerItem(258, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_26.addItem(spacerItem18)
        self.pushButton_get_result = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_get_result.setMinimumSize(QtCore.QSize(80, 40))
        self.pushButton_get_result.setObjectName("pushButton_get_result")
        self.pushButton_get_result.clicked.connect(self.on_click_get_result)
        self.horizontalLayout_26.addWidget(self.pushButton_get_result)
        spacerItem19 = QtWidgets.QSpacerItem(258, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_26.addItem(spacerItem19)
        self.verticalLayout_6.addLayout(self.horizontalLayout_26)
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        spacerItem20 = QtWidgets.QSpacerItem(148, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_27.addItem(spacerItem20)
        self.label_result = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_result.setFont(font)
        self.label_result.setObjectName("label_result")
        self.horizontalLayout_27.addWidget(self.label_result)
        spacerItem21 = QtWidgets.QSpacerItem(148, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_27.addItem(spacerItem21)
        self.verticalLayout_6.addLayout(self.horizontalLayout_27)
        self.verticalLayout_8.addLayout(self.verticalLayout_6)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.full_data_ = full_data.copy()

    def on_click_get_file(self):
        file, _ = QFileDialog.getOpenFileName(None, 'Выбрать файл с исходными данными для нейросети', './', "Dataset (*.csv *.xls)")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Новые данные для модели")
        msg.setText("Исходные данные для обучения модели успешно изменены!")
        retval = msg.exec_()

    def on_click_get_result(self):
        pump_id = self.lineEdit_pump_id.text().replace(",", ".")
        Q_gas_TM = self.lineEdit_Q_gas_TM.text().replace(",", ".")
        Q_liq_TM = self.lineEdit_Q_liq_TM.text().replace(",", ".")
        Q_liq = self.lineEdit_Q_liq.text().replace(",", ".")
        W_act = self.lineEdit_W_act.text().replace(",", ".")
        OBV_HAL = self.lineEdit_OBV_HAL.text().replace(",", ".")
        N_d = self.lineEdit_N_d.text().replace(",", ".")
        P_zatr = self.lineEdit_P_zatr.text().replace(",", ".")
        P_buf = self.lineEdit_P_buf.text().replace(",", ".")
        D = self.lineEdit_D.text().replace(",", ".")
        P_lin = self.lineEdit_P_lin.text().replace(",", ".")
        KVCH_HAL = self.lineEdit_KVCH_HAL.text().replace(",", ".")
        I = self.lineEdit_I.text().replace(",", ".")
        F = self.lineEdit_F.text().replace(",", ".")
        Q_n = self.lineEdit_Q_n.text().replace(",", ".")

        if not pump_id.isdigit():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение № насоса должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(Q_gas_TM):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Qгаз тм должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(Q_liq_TM):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Qж тм должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(Q_liq):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Qж должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(W_act):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Wакт должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(OBV_HAL):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Обв-ХАЛ должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(N_d):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Нд должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(P_zatr):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Pзатр должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(P_buf):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Pбуф должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(D):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Dшт должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(P_lin):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Pлин должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(KVCH_HAL):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение КВЧ-ХАЛ должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(I):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение I должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(F):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение F должно быть численным!")
            retval = msg.exec_()
        elif not is_digit(Q_n):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка при вводе данных")
            msg.setText("Некорректные данные.\nЗначение Qн должно быть численным!")
            retval = msg.exec_()
        else:
            test = [float(self.lineEdit_Q_gas_TM.text().replace(",", ".")),
                    float(self.lineEdit_Q_liq_TM.text().replace(",", ".")),
                    float(self.lineEdit_Q_liq.text().replace(",", ".")),
                    float(self.lineEdit_W_act.text().replace(",", ".")),
                    float(self.lineEdit_OBV_HAL.text().replace(",", ".")),
                    float(self.lineEdit_N_d.text().replace(",", ".")),
                    float(self.lineEdit_P_zatr.text().replace(",", ".")),
                    float(self.lineEdit_P_buf.text().replace(",", ".")),
                    float(self.lineEdit_D.text().replace(",", ".")),
                    float(self.lineEdit_P_lin.text().replace(",", ".")),
                    float(self.lineEdit_KVCH_HAL.text().replace(",", ".")),
                    float(self.lineEdit_I.text().replace(",", ".")),
                    float(self.lineEdit_F.text()), float(self.lineEdit_Q_n.text().replace(",", ".")),
                    int(self.lineEdit_pump_id.text())]

            full_data_, pm = add_row(self.full_data_, test)
            probability = round(raw_predict(pm, std_scaler_final, model_final) * 100, 2)
            self.label_result.setText("Вероятность отказа составляет " + str(probability) + "%")

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Результат")
            msg.setText("По заданным параметрам значение вероятности отказа составляет: " + str(probability) + "%")
            retval = msg.exec_()

    def on_click_save_result(self):
        name, _ = QFileDialog.getSaveFileName(None, "Сохранить результат", './', "Текстовый документ (*.txt)")
        file = open(name, "w")
        file.write("____________________________________\n")
        file.write("Получение вероятности отказа        |\n")
        file.write("____________________________________|\n")
        file.write("Исходные данные:\n")
        file.write("Номер установки: " + str(self.lineEdit_pump_id.text().replace(",",".")) + "\n")
        file.write("Qгаз тм: " + str(self.lineEdit_Q_gas_TM.text().replace(",", ".")) + "\n")
        file.write("Qж тм: " + str(self.lineEdit_Q_liq_TM.text().replace(",", ".")) + "\n")
        file.write("Qж: " + str(self.lineEdit_Q_liq.text().replace(",", ".")) + "\n")
        file.write("Wакт: " + str(self.lineEdit_W_act.text().replace(",", ".")) + "\n")
        file.write("Обв-ХАЛ: " + str(self.lineEdit_OBV_HAL.text().replace(",", ".")) + "\n")
        file.write("Нд: " + str(self.lineEdit_N_d.text().replace(",", ".")) + "\n")
        file.write("Pзатр: " + str(self.lineEdit_P_zatr.text().replace(",", ".")) + "\n")
        file.write("Pбуф: " + str(self.lineEdit_P_buf.text().replace(",", ".")) + "\n")
        file.write("Dшт: " + str(self.lineEdit_D.text().replace(",", ".")) + "\n")
        file.write("Pлин: " + str(self.lineEdit_P_lin.text().replace(",", ".")) + "\n")
        file.write("КВЧ-ХАЛ: " + str(self.lineEdit_KVCH_HAL.text().replace(",", ".")) + "\n")
        file.write("I: " + str(self.lineEdit_I.text().replace(",", ".")) + "\n")
        file.write("F: " + str(self.lineEdit_F.text().replace(",", ".")) + "\n")
        file.write("Qн: " + str(self.lineEdit_Q_n.text().replace(",", ".")) + "\n")
        file.write("____________________________________\n")
        file.write(str(self.label_result.text()))
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Запись в файл")
        msg.setText("Результаты успешно записаны в файл " + name + ".")
        retval = msg.exec_()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Вероятность отказа УЭЦН"))
        self.pushButton_get_file.setText(_translate("MainWindow", "Изменить данные\nдля обучения модели"))
        self.pushButton_save_result.setText(_translate("MainWindow", "Сохранить\nрезультаты"))
        self.label_pump_id.setText(_translate("MainWindow", "№ насоса"))
        self.lineEdit_pump_id.setText(_translate("MainWindow", "488"))
        self.label_Q_gas_TM.setText(_translate("MainWindow", "Qгаз тм"))
        self.lineEdit_Q_gas_TM.setText(_translate("MainWindow", "3.1684"))
        self.label_Q_liq_TM.setText(_translate("MainWindow", "Qж тм"))
        self.lineEdit_Q_liq_TM.setText(_translate("MainWindow", "0.2526"))
        self.label_Q_liq.setText(_translate("MainWindow", "Qж"))
        self.lineEdit_Q_liq.setText(_translate("MainWindow", "14.5"))
        self.label_W_act.setText(_translate("MainWindow", "Wакт"))
        self.lineEdit_W_act.setText(_translate("MainWindow", "1.5601"))
        self.label_OBV_HAL.setText(_translate("MainWindow", "Обв-ХАЛ"))
        self.lineEdit_OBV_HAL.setText(_translate("MainWindow", "38.8499"))
        self.label_N_d.setText(_translate("MainWindow", "Hд"))
        self.lineEdit_N_d.setText(_translate("MainWindow", "669.1428"))
        self.label_P_zatr.setText(_translate("MainWindow", "Pзатр"))
        self.lineEdit_P_zatr.setText(_translate("MainWindow", "3"))
        self.label_P_buf.setText(_translate("MainWindow", "Pбуф"))
        self.lineEdit_P_buf.setText(_translate("MainWindow", "2.5714"))
        self.label_D.setText(_translate("MainWindow", "Dшт"))
        self.lineEdit_D.setText(_translate("MainWindow", "12"))
        self.label_P_lin.setText(_translate("MainWindow", "Pлин"))
        self.lineEdit_P_lin.setText(_translate("MainWindow", "4.6666"))
        self.label_KVCH_HAL.setText(_translate("MainWindow", "КВЧ-ХАЛ"))
        self.lineEdit_KVCH_HAL.setText(_translate("MainWindow", "5.14285"))
        self.label_I.setText(_translate("MainWindow", "I"))
        self.lineEdit_I.setText(_translate("MainWindow", "8.3333"))
        self.label_F.setText(_translate("MainWindow", "F"))
        self.lineEdit_F.setText(_translate("MainWindow", "33.3333"))
        self.label_Q_n.setText(_translate("MainWindow", "Qн"))
        self.lineEdit_Q_n.setText(_translate("MainWindow", "2.7614"))
        self.pushButton_get_result.setText(_translate("MainWindow", "Получить\nвероятность"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())