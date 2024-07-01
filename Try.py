import pandas as pd
import tensorflow as tf
import joblib


custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'mae': tf.keras.metrics.MeanAbsoluteError(),
    'Adam': tf.keras.optimizers.Adam
}

model = tf.keras.models.load_model('my_model.keras', custom_objects=custom_objects)


ct = joblib.load('column_transformer.pkl')


koko = pd.DataFrame([[7.916944, 240.128979]], columns=['Global_intensity', 'Voltage'])


koko_standardized = ct.transform(koko)

predictions = model.predict(koko_standardized)

print("Predictions:", predictions)
