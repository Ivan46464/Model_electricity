import re

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('../household_power_consumption.csv', delimiter=',', parse_dates={'DateTime': ['Date', 'Time']})


df_clean = df.dropna().copy()


df_clean['DateTime'] = pd.to_datetime(df_clean['DateTime'])

def clean_and_convert(column):
    cleaned_column = column.apply(lambda x: re.sub(r'[^\d.]', '', x)).astype(float)
    return cleaned_column


df_clean['Voltage'] = clean_and_convert(df_clean['Voltage'])
df_clean['Global_intensity'] = clean_and_convert(df_clean['Global_intensity'])
df_clean['Global_active_power'] = clean_and_convert(df_clean['Global_active_power'])
df_clean['Global_reactive_power'] = clean_and_convert(df_clean['Global_reactive_power'])
df_clean['Sub_metering_1'] = clean_and_convert(df_clean['Sub_metering_1'])
df_clean['Sub_metering_2'] = clean_and_convert(df_clean['Sub_metering_2'])


df_daily = df_clean.resample('D', on='DateTime').mean().reset_index()
df_daily.fillna(0, inplace=True)
print(df_daily.loc[0, :])

X = df_daily[['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3']]
y = df_daily['Sub_metering_1']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#joblib.dump(scaler, 'scaler_sub_1.pkl')

model_knn = KNeighborsRegressor(metric= 'manhattan', n_neighbors=7, weights='uniform')

model_knn.fit(X_train_scaled, y_train)
#joblib.dump(model_knn, 'best_knn_model_sub_1.pkl')


y_pred_knn = model_knn.predict(X_test_scaled)
val_mae_knn = mean_absolute_error(y_test, y_pred_knn)

print("KNN Regression R² Score: ", model_knn.score(X_test_scaled, y_test))
print("KNN Regression Final MAE: ", val_mae_knn)


plt.scatter(y_test, y_pred_knn, alpha=0.3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('KNN Regression Predictions')
plt.show()

for i in range(len(y_test)):
    print("Actual: " + str(y_test.iloc[i]) + " and predicted: " + str(y_pred_knn[i]))
print(len(y_test))