import joblib
import pandas as pd



loaded_model = joblib.load('best_knn_model_sub_1.pkl')


scaler = joblib.load('scaler_sub_1.pkl')


koko = pd.DataFrame([[2.102893, 7.916944, 0.244444, 4.0833]], columns=['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3'])


koko_scaled = scaler.transform(koko)


predictions = loaded_model.predict(koko_scaled)

print("Predictions:", predictions)
