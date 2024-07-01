import joblib
import pandas as pd

loaded_model = joblib.load('best_knn_model_sub_2.pkl')

scaler = joblib.load('scaler_sub_2.pkl')

koko = pd.DataFrame([[0.102893, 7.916944, 0.0, 4.0833]], columns=['Global_active_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_3'])

koko_scaled = scaler.transform(koko)

predictions = loaded_model.predict(koko_scaled)

print("Predictions:", predictions)
