import pandas as pd
from pycaret.regression import RegressionExperiment
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Volve P-12_DatesCorrected.csv")

# Rename the 'Unnamed: 0' column to 'Date' and convert it to datetime format
_ = df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Count occurrences of AVG_DOWNHOLE_TEMPERATURE being 0
zero_temperature_count = df[df['AVG_DOWNHOLE_TEMPERATURE'] == 0]['AVG_DOWNHOLE_TEMPERATURE'].count()

# Filter data before 2011-01-01 and where AVG_DOWNHOLE_TEMPERATURE is not 0
df_before_2010 = df[(df['Date'] <= '2011-01-01') & (df['AVG_DOWNHOLE_TEMPERATURE'] != 0)]

# Display the maximum date in the filtered data
max_date_before_2010 = df_before_2010['Date'].max()

# Plot a scatter plot of AVG_DOWNHOLE_TEMPERATURE vs. Date before 2010
plt.scatter(df_before_2010['Date'], df_before_2010['AVG_DOWNHOLE_TEMPERATURE'], color='green')
plt.xlabel('Date')
plt.ylabel('AVG_DOWNHOLE_TEMPERATURE')
plt.title('Scatter Plot of Temperature vs Date (Before 2010)')
plt.ylim(100, 110)
plt.show()

# Initialize a RegressionExperiment
exp1 = RegressionExperiment()

# Setup the experiment with data before 2010 and target variable 'AVG_DOWNHOLE_TEMPERATURE'
exp1.setup(df_before_2010, target='AVG_DOWNHOLE_TEMPERATURE')

# Compare and select the best regression model
best_model_before_2010 = exp1.compare_models()

# Evaluate the performance of the selected model
exp1.evaluate_model(best_model_before_2010)

# Filter data after the maximum date before 2010 and drop the target variable
df_after_2010 = df[df['Date'] > max_date_before_2010].drop(columns=['AVG_DOWNHOLE_TEMPERATURE'])

# Display the minimum date in the filtered data after 2010
min_date_after_2010 = df_after_2010['Date'].min()

# Make predictions on the data after 2010 using the best model
predictions_after_2010 = exp1.predict_model(best_model_before_2010, data=df_after_2010)

# Display summary statistics of the prediction labels
prediction_label_stats = predictions_after_2010['prediction_label'].describe()

# Plot a scatter plot of actual vs. predicted AVG_DOWNHOLE_TEMPERATURE after 2010
fig, ax = plt.subplots()
df_before_2010.plot(x='Date', y='AVG_DOWNHOLE_TEMPERATURE', kind='scatter', color='blue', ax=ax, label='Actual')
predictions_after_2010.plot(x='Date', y='prediction_label', kind='scatter', color='orange', ax=ax, label='Prediction')
plt.ylim(100, 110)
plt.legend()
plt.title('Actual vs Predicted Temperature (After 2010)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()
