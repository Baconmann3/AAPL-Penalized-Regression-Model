import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

print("===== Step 1: Loading and Preparing Data =====")
# Define the file path
file_path = '/Users/shaunakseth/Desktop/top_10_stock_data.csv'

# Load the CSV, parse dates, filter for AAPL, and sort by date
df = pd.read_csv(file_path, parse_dates=['Date'])
df = df[df['Ticker'] == 'AAPL']
df = df.sort_values('Date')

# Create target variable (next day's closing price)
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

# Extract year and month from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

print(f"Data spans from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

# Split data into training (2015-2018) and 2019 for testing
training_data = df[df['Year'] <= 2018]
data_2019 = df[df['Year'] == 2019]

print(f"\nTraining data (2015-2018): {len(training_data)} rows")
print(f"2019 data: {len(data_2019)} rows")
print(f"Trading days by month in 2019:")
print(data_2019.groupby('Month').size())

# Create features and targets
X_train = training_data[['Open', 'High', 'Low', 'Close', 'Volume']]
y_train = training_data['Target']
X_test = data_2019[['Open', 'High', 'Low', 'Close', 'Volume']]
y_test = data_2019['Target']

print("\n===== Step 2: Training the Model =====")
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95]
}

# Set up cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize ElasticNet model
model = ElasticNet(random_state=42, max_iter=10000)

# Set up grid search
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Train the model
print("Training model with grid search...")
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

print("\n===== Step 3: Full 2019 Forward Prediction =====")
# Get the last day of 2018 as starting point
last_2018_data = training_data.iloc[-1]
print(f"Last trading day of 2018: {last_2018_data['Date'].strftime('%Y-%m-%d')}")
print(f"Last closing price of 2018: ${last_2018_data['Close']:.2f}")

# Get 2019 trading days
days_2019 = sorted(data_2019['Date'])
print(f"Number of trading days in 2019: {len(days_2019)}")
print(f"First trading day of 2019: {days_2019[0].strftime('%Y-%m-%d')}")
print(f"Last trading day of 2019: {days_2019[-1].strftime('%Y-%m-%d')}")

# Perform forward prediction
print("\nRunning forward prediction simulation...")
# Initial state from last trading day of 2018
current_open = last_2018_data['Open']
current_high = last_2018_data['High']
current_low = last_2018_data['Low']
current_close = last_2018_data['Close']
current_volume = last_2018_data['Volume']

# Container for forward predictions
forward_predictions = []

# Perform forward prediction for 2019
for date in days_2019:
    # Create features for current state
    features = np.array([[current_open, current_high, current_low, current_close, current_volume]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict next close
    next_close_pred = best_model.predict(features_scaled)[0]

    # Store prediction with date
    forward_predictions.append(next_close_pred)

    # Get actual data for this date to update state
    actual_day_data = data_2019[data_2019['Date'] == date].iloc[0]

    # Update state for next day (using actual open, high, low, but predicted close)
    # This simulates having the day's trading data but not the closing price
    current_open = actual_day_data['Open']
    current_high = actual_day_data['High']
    current_low = actual_day_data['Low']
    current_close = next_close_pred  # Use predicted close
    current_volume = actual_day_data['Volume']

print("\n===== Step 4: Comparing Predictions with Actual Data =====")
# Create DataFrame for comparison
comparison_df = pd.DataFrame({
    'Date': days_2019,
    'Actual_Close': data_2019['Close'].values,
    'Predicted_Close': forward_predictions
})

# Add month information for monthly analysis
comparison_df['Year'] = comparison_df['Date'].dt.year
comparison_df['Month'] = comparison_df['Date'].dt.month
comparison_df['MonthName'] = comparison_df['Date'].dt.strftime('%b')
comparison_df['Quarter'] = comparison_df['Date'].dt.quarter
comparison_df['Error'] = comparison_df['Actual_Close'] - comparison_df['Predicted_Close']
comparison_df['Error_Pct'] = (comparison_df['Error'] / comparison_df['Actual_Close']) * 100
comparison_df['Abs_Error_Pct'] = np.abs(comparison_df['Error_Pct'])

# Calculate overall metrics
rmse = np.sqrt(mean_squared_error(comparison_df['Actual_Close'], comparison_df['Predicted_Close']))
mae = mean_absolute_error(comparison_df['Actual_Close'], comparison_df['Predicted_Close'])
mape = np.mean(comparison_df['Abs_Error_Pct'])
r2 = r2_score(comparison_df['Actual_Close'], comparison_df['Predicted_Close'])

print("\nOverall Prediction Performance (2019):")
print(f"RMSE: ${rmse:.4f}")
print(f"MAE: ${mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")

# Calculate quarterly metrics
quarterly_metrics = []
for quarter in range(1, 5):
    q_data = comparison_df[comparison_df['Quarter'] == quarter]
    q_rmse = np.sqrt(mean_squared_error(q_data['Actual_Close'], q_data['Predicted_Close']))
    q_mae = mean_absolute_error(q_data['Actual_Close'], q_data['Predicted_Close'])
    q_mape = np.mean(q_data['Abs_Error_Pct'])
    q_r2 = r2_score(q_data['Actual_Close'], q_data['Predicted_Close'])

    quarterly_metrics.append({
        'Quarter': f"Q{quarter}",
        'Days': len(q_data),
        'RMSE': q_rmse,
        'MAE': q_mae,
        'MAPE': q_mape,
        'R²': q_r2,
        'Start_Actual': q_data['Actual_Close'].iloc[0],
        'End_Actual': q_data['Actual_Close'].iloc[-1],
        'Actual_Return': (q_data['Actual_Close'].iloc[-1] / q_data['Actual_Close'].iloc[0] - 1) * 100,
        'Predicted_Return': (q_data['Predicted_Close'].iloc[-1] / q_data['Predicted_Close'].iloc[0] - 1) * 100
    })

quarterly_metrics_df = pd.DataFrame(quarterly_metrics)
print("\nQuarterly Performance Metrics:")
print(quarterly_metrics_df[['Quarter', 'RMSE', 'MAE', 'MAPE', 'R²']].to_string(index=False))

print("\nQuarterly Price Movements:")
for _, row in quarterly_metrics_df.iterrows():
    print(
        f"{row['Quarter']}: Actual ${row['Start_Actual']:.2f} → ${row['End_Actual']:.2f} ({row['Actual_Return']:.1f}%), " +
        f"Predicted {row['Predicted_Return']:.1f}%")

# Calculate monthly metrics (useful for the visualization)
monthly_metrics = []
for month in range(1, 13):
    if month in comparison_df['Month'].values:  # Check if we have data for this month
        month_data = comparison_df[comparison_df['Month'] == month]
        month_rmse = np.sqrt(mean_squared_error(month_data['Actual_Close'], month_data['Predicted_Close']))
        month_mae = mean_absolute_error(month_data['Actual_Close'], month_data['Predicted_Close'])
        month_mape = np.mean(month_data['Abs_Error_Pct'])
        month_r2 = r2_score(month_data['Actual_Close'], month_data['Predicted_Close'])

        monthly_metrics.append({
            'Month': month,
            'MonthName': month_data['MonthName'].iloc[0],
            'Days': len(month_data),
            'RMSE': month_rmse,
            'MAE': month_mae,
            'MAPE': month_mape,
            'R²': month_r2,
            'Start_Actual': month_data['Actual_Close'].iloc[0],
            'End_Actual': month_data['Actual_Close'].iloc[-1],
            'Actual_Return': (month_data['Actual_Close'].iloc[-1] / month_data['Actual_Close'].iloc[0] - 1) * 100,
            'Predicted_Return': (month_data['Predicted_Close'].iloc[-1] / month_data['Predicted_Close'].iloc[
                0] - 1) * 100
        })

monthly_metrics_df = pd.DataFrame(monthly_metrics)

print("\n===== Step 5: Visualizing Full 2019 Predictions =====")
# Create visualization
plt.figure(figsize=(18, 10))

# Plot prices
plt.subplot(2, 1, 1)
plt.plot(comparison_df['Date'], comparison_df['Actual_Close'], 'b-', linewidth=2, label='Actual Closing Price')
plt.plot(comparison_df['Date'], comparison_df['Predicted_Close'], 'r--', linewidth=1.5, label='Predicted Closing Price')

# Add quarter separators
for quarter in range(2, 5):
    quarter_start = comparison_df[comparison_df['Quarter'] == quarter]['Date'].iloc[0]
    plt.axvline(x=quarter_start, color='gray', linestyle='-', alpha=0.3)
    plt.text(quarter_start, plt.ylim()[1] * 0.95, f"Q{quarter}", ha='center', va='top', backgroundcolor='white')

plt.title('AAPL Stock: 2019 Actual vs. Predicted Closing Prices', fontsize=14)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Format x-axis with month labels
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Plot error
plt.subplot(2, 1, 2)
plt.bar(comparison_df['Date'], comparison_df['Error'], color=['g' if x >= 0 else 'r' for x in comparison_df['Error']],
        alpha=0.6, width=1.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add quarter separators
for quarter in range(2, 5):
    quarter_start = comparison_df[comparison_df['Quarter'] == quarter]['Date'].iloc[0]
    plt.axvline(x=quarter_start, color='gray', linestyle='-', alpha=0.3)

plt.title('Prediction Error (Actual - Predicted)', fontsize=14)
plt.ylabel('Error ($)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Format x-axis with month labels
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Add average error line
avg_error = comparison_df['Error'].mean()
plt.axhline(y=avg_error, color='blue', linestyle='--', linewidth=1.5, label=f'Avg Error: ${avg_error:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('aapl_2019_prediction.png', dpi=300)
plt.show()

# Plot cumulative returns
plt.figure(figsize=(16, 8))

# Calculate cumulative returns
comparison_df['Actual_Cum_Return'] = comparison_df['Actual_Close'] / comparison_df['Actual_Close'].iloc[0] - 1
comparison_df['Predicted_Cum_Return'] = comparison_df['Predicted_Close'] / comparison_df['Predicted_Close'].iloc[0] - 1

plt.plot(comparison_df['Date'], comparison_df['Actual_Cum_Return'] * 100, 'b-', linewidth=2, label='Actual')
plt.plot(comparison_df['Date'], comparison_df['Predicted_Cum_Return'] * 100, 'r--', linewidth=1.5, label='Predicted')

# Add quarter separators
for quarter in range(2, 5):
    quarter_start = comparison_df[comparison_df['Quarter'] == quarter]['Date'].iloc[0]
    plt.axvline(x=quarter_start, color='gray', linestyle='-', alpha=0.3)
    plt.text(quarter_start, plt.ylim()[1] * 0.95, f"Q{quarter}", ha='center', va='top', backgroundcolor='white')

plt.title('AAPL Stock: 2019 Cumulative Returns (%)', fontsize=14)
plt.ylabel('Cumulative Return (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Format x-axis with month labels
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Add final return labels
plt.annotate(f"{comparison_df['Actual_Cum_Return'].iloc[-1] * 100:.1f}%",
             xy=(comparison_df['Date'].iloc[-1], comparison_df['Actual_Cum_Return'].iloc[-1] * 100),
             xytext=(5, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

plt.annotate(f"{comparison_df['Predicted_Cum_Return'].iloc[-1] * 100:.1f}%",
             xy=(comparison_df['Date'].iloc[-1], comparison_df['Predicted_Cum_Return'].iloc[-1] * 100),
             xytext=(5, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

plt.tight_layout()
plt.savefig('aapl_2019_returns.png', dpi=300)
plt.show()

# Create quarterly performance visualization
plt.figure(figsize=(14, 15))

# Plot 1: Quarterly MAPE
plt.subplot(3, 1, 1)
bars = plt.bar(quarterly_metrics_df['Quarter'], quarterly_metrics_df['MAPE'], color='skyblue')
plt.axhline(y=mape, color='red', linestyle='--', label=f'Overall MAPE: {mape:.2f}%')
plt.title('Prediction Error by Quarter (MAPE)', fontsize=14)
plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

# Add value labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: Quarterly Returns Comparison
plt.subplot(3, 1, 2)
x = np.arange(len(quarterly_metrics_df))
width = 0.35
plt.bar(x - width / 2, quarterly_metrics_df['Actual_Return'], width, label='Actual', color='blue', alpha=0.7)
plt.bar(x + width / 2, quarterly_metrics_df['Predicted_Return'], width, label='Predicted', color='red', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.title('Quarterly Returns: Actual vs. Predicted', fontsize=14)
plt.ylabel('Quarterly Return (%)', fontsize=12)
plt.xticks(x, quarterly_metrics_df['Quarter'])
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

# Add value labels above/below each bar
for i, v in enumerate(quarterly_metrics_df['Actual_Return']):
    plt.text(i - width / 2, v + (1 if v >= 0 else -2), f"{v:.1f}%", ha='center', fontsize=9)
for i, v in enumerate(quarterly_metrics_df['Predicted_Return']):
    plt.text(i + width / 2, v + (1 if v >= 0 else -2), f"{v:.1f}%", ha='center', fontsize=9)

# Plot 3: Monthly MAPE
plt.subplot(3, 1, 3)
bars = plt.bar(monthly_metrics_df['MonthName'], monthly_metrics_df['MAPE'], color='lightgreen')
plt.axhline(y=mape, color='red', linestyle='--', label=f'Overall MAPE: {mape:.2f}%')
plt.title('Prediction Error by Month (MAPE)', fontsize=14)
plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

# Add value labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('aapl_2019_quarterly_analysis.png', dpi=300)
plt.show()

# Create a heatmap to visualize error patterns throughout the year
plt.figure(figsize=(16, 8))

# Pivot the data to create a heatmap of errors by day/month
comparison_df['Day'] = comparison_df['Date'].dt.day
comparison_df['MonthDay'] = comparison_df['Date'].dt.strftime('%b-%d')

# Create a pivot table for the heatmap
pivot_data = comparison_df.pivot_table(
    index='Month',
    columns='Day',
    values='Error_Pct',
    aggfunc='mean'
)

# Create labels for the x and y axes
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_labels = range(1, 32)

# Create heatmap
plt.imshow(pivot_data, cmap='RdYlGn_r', aspect='auto', vmin=-15, vmax=15)
plt.colorbar(label='Error Percentage (%)')
plt.title('Prediction Error Heatmap by Day of Month (2019)', fontsize=14)
plt.yticks(range(len(month_labels)), month_labels)
plt.xticks(range(0, 31, 5), [str(d) for d in range(1, 32, 5)])
plt.xlabel('Day of Month', fontsize=12)
plt.ylabel('Month', fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig('aapl_2019_error_heatmap.png', dpi=300)
plt.show()

print("\n===== Analysis Complete =====")
print(
    "Images saved: 'aapl_2019_prediction.png', 'aapl_2019_returns.png', 'aapl_2019_quarterly_analysis.png', 'aapl_2019_error_heatmap.png'")