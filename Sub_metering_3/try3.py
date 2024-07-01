import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

loaded_model = joblib.load('best_knn_model.pkl')


koko = pd.DataFrame([[1.7731333333333337, 7.3]], columns=['Global_active_power', 'Global_intensity'])


scaler = StandardScaler()
scaler = joblib.load('scaler.pkl')


koko_scaled = scaler.transform(koko)


predictions = loaded_model.predict(koko_scaled)

print("Predictions:", predictions)