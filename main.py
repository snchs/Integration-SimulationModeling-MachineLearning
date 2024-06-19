import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Activation
from tensorflow.keras.optimizers import Adam

# Кастомная активационная функция
def custom_activation(x):
    return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)


# Кастомный слой
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)


# Уникальная архитектура нейронной сети
def build_custom_model(input_shape):
    inputs = Input(shape=input_shape)

    # Первый блок
    x1 = Dense(200)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation(custom_activation)(x1)
    x1 = Dropout(0.3)(x1)

    # Второй блок
    x2 = Dense(64)(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.3)(x2)

    # Третий блок
    x3 = CustomLayer(1024)(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)

    # Четвертый блок с пропусками связей (Skip connections)
    x4 = Dense(16, activation='relu')(x3)
    x4 = Concatenate()([x4, x1])  # Skip connection

    outputs = Dense(1)(x4)

    model = Model(inputs, outputs)
    return model


# Загрузка данных
data = pd.read_csv('resultDataSet.csv', delimiter=';')
X = data.iloc[:, 1:6].values  # Фичи
y = data.iloc[:, 6].values  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=192)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Построение и компиляция модели
model = build_custom_model(input_shape=(X_train.shape[1],))
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Обучение модели
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)


# Оценка модели на тестовых данных
def evaluate_model(model, X_test, y_test):


    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'R^2 Score: {r2}')


    # Визуализация предсказаний
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Value')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Value')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.grid(True)
    # plt.show()

    return y_pred



# Вызов функции оценки модели
y_pred = evaluate_model(model, X_test, y_test)
model.save('DS_NN_Model.h5')
# График потерь (Loss) во время обучения
plt.figure(figsize=(14, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# plt.show()



