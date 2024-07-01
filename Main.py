
import pandas as pd

df = pd.read_csv('hourly_power_consumption.csv', delimiter=',')
df.drop(columns=['index', 'DateTime'], inplace=True)
df_first_120 = df.head(120)
# Create the reports dictionary
reports = {}
for i, row in df_first_120.iterrows():
    report_name = f"report{i+1}"
    reports[report_name] = {
        "Global_active_power": row['Global_active_power'],
        "Global_reactive_power": row['Global_reactive_power'],
        "Voltage": row['Voltage'],
        "Global_intensity": row['Global_intensity'],
        "Sub_metering_1": row['Sub_metering_1'],
        "Sub_metering_2": row['Sub_metering_2'],
        "Sub_metering_3": row['Sub_metering_3']
    }

print(reports)

