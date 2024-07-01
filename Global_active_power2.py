import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib


tf.random.set_seed(35)


df = pd.read_csv('household_power_consumption.csv', delimiter=',')


df_clean = df.dropna()


df_clean = df_clean.drop(['Date', 'Time', 'index','Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1, errors='ignore')


df_clean['Global_active_power'] = df_clean['Global_active_power'].astype('float')
df_clean['Voltage'] = df_clean['Voltage'].astype('float')
df_clean['Global_intensity'] = df_clean['Global_intensity'].astype('float')


features = df_clean[['Global_intensity', 'Voltage']]
labels = df_clean['Global_active_power']


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)


ct = ColumnTransformer([('standardize', StandardScaler(), ['Global_intensity', 'Voltage'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)


joblib.dump(ct, 'column_transformer.pkl')


model = Sequential(name="my_model")
num_features = features.shape[1]
model.add(InputLayer(input_shape=(num_features,)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


opt = Adam(learning_rate=0.001)
model.compile(loss='mse', metrics=['mae'], optimizer=opt)


print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


history = model.fit(features_train, labels_train, epochs=300, batch_size=20000, verbose=1, validation_data=(features_test, labels_test), callbacks=[early_stopping])


model.save('my_model.keras')


val_mse, val_mae = model.evaluate(features_test, labels_test, verbose=0)
print("Final MAE: ", val_mae)


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

