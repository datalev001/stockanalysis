import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve
import shap
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import yfinance as yf

## download stock data
def get_data_by_day(ticker_list, start, end):
    interval = '1d'
    # Predefine columns for the DataFrame to avoid appending data in the loop
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
    data_list = []
    for ticker in ticker_list:
        df = yf.download(ticker, start=start, end=end, interval=interval)
        if not df.empty:
            df['ticker'] = ticker
            data_list.append(df)
    download_data = pd.concat(data_list).reset_index()
    download_data = download_data[columns]  # Reorder and select relevant columns
    
    unique_dates = download_data['Date'].drop_duplicates().sort_values(ascending=False).reset_index(drop=True)
    date_map = pd.Series(range(1, len(unique_dates) + 1), index=unique_dates)
    download_data['dayseq'] = download_data['Date'].map(date_map)
    download_data = download_data.sort_values(['ticker', 'Date'], ascending=[False, False])
    return download_data

'''
example:

stocks = tks2 = ['SPY', 'QQQ', 'DIA', 'SBUX', 'AAPL', 'GOOG', 'FB','ASML', 'SSO', 'DIA', 'SYK', 'EMB', 'MU', 'AEE', 'TRU',\
'BAC', 'JNJ', 'TQQQ', 'BA', 'JPM', 'TSLA',  'CRM', 'CAT', 'DE', 'ALGN', 'ADP', 'AN', 'EXP', 'MDB',
 'CRWD',  'DHI',  'FIVN',  'HD']

startday = '2022-06-01'
latestday = '2023-06-01'
df = get_data_by_day(tks, startday, latestday)

'''


## data cleaning
# Adjusting column names for consistency
df = df[['ticker', 'date_dt', 'Close', 'Volume']].rename(columns={'date_dt': 'date', 'Close': 'price', 'Volume': 'Volume'})
# Removing stocks with prices below $1
min_prices = df.groupby('ticker')['price'].min().reset_index()
df = df[~df.ticker.isin(min_prices[min_prices.price < 1].ticker)]
# Eliminating stocks with non-positive volumes
min_volumes = df.groupby('ticker')['Volume'].min().reset_index()
df = df[~df.ticker.isin(min_volumes[min_volumes.Volume < 1].ticker)]


### make data smooth by spline
# Assuming 'df' is your DataFrame with columns: ['ticker', 'date', 'price', 'Volume']
result_df = pd.DataFrame()
for ticker in df['ticker'].unique():
    df_ticker = df[df['ticker'] == ticker].sort_values(by='date')
    # Generate x values and apply cubic spline interpolation to price and volume
    x_vals = np.arange(len(df_ticker))
    spline_price = make_interp_spline(x_vals, df_ticker['price'], k=3)(np.linspace(x_vals.min(), x_vals.max(), 300))
    spline_volume = make_interp_spline(x_vals, df_ticker['Volume'], k=3)(np.linspace(x_vals.min(), x_vals.max(), 300))

    # Select 30 evenly spaced points from the smoothed curves
    indices = np.linspace(0, 299, 30, dtype=int)
    # Prepare the smoothed data DataFrame for this ticker
    df_smooth = pd.DataFrame({
        'ticker': ticker,
        'smooth_price': spline_price[indices],
        'smooth_volume': spline_volume[indices],
        'seq': np.arange(30, 0, -1)  # Descending sequence from 30 to 1})

# Append the smoothed data
result_df = pd.concat([result_df, df_smooth], ignore_index=True)


###kmeans clustering stocks
# Data pivoting
result_df_pivoted = result_df.pivot(index='ticker', columns='seq', values=['smooth_price', 'smooth_volume'])
result_df_pivoted.columns = [f'{val}_{i}' for val, i in result_df_pivoted.columns]
result_df_pivoted.reset_index(inplace=True)

# Data standardization
scaler = StandardScaler()
price_columns = [col for col in result_df_pivoted.columns if 'smooth_price' in col]
volume_columns = [col for col in result_df_pivoted.columns if 'smooth_volume' in col]
result_df_pivoted[price_columns] = scaler.fit_transform(result_df_pivoted[price_columns])
result_df_std = result_df_pivoted[['ticker'] + price_columns]

# Determining the optimal number of clusters
K = range(1, 10)
sum_of_squared_distances = []
for k in K:
    kmeans = KMeans(n_clusters=k).fit(result_df_std.drop('ticker', axis=1))
    sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('K values')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal K')
plt.show()

## create clusters and evaluate performance by Silhouette Score
kmeans = KMeans(n_clusters=4)
result_df_std['cluster'] = kmeans.fit_predict(result_df_std.drop('ticker', axis=1))
# Performance evaluation
silhouette_score = metrics.silhouette_score(result_df_std.drop(['ticker', 'cluster'], axis=1), result_df_std['cluster'])
print(f'Silhouette Score: {silhouette_score}')

# Apply PCA to reduce dimensionality for visualizing the cluster distribution
from sklearn.decomposition import PCA
# Initialize PCA and reduce X to two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(result_df_std[cols])
# Plotting the PCA-transformed features colored by segment
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.75, s=7)
plt.title('PCA of Stock Tickers by Segment')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Segment')
plt.show()



# organize training dara and feature generation and engineering
df0 = result_df_pivoted[['ticker'] + volume_columns + price_columns]
volume_mean = df0[volume_columns].mean(axis= 1) 
for it in volume_columns:
    df0[it] = df0[it] / volume_mean
    
volume_price = df0[price_columns].mean(axis= 1)     
for it in price_columns:
    df0[it] = df0[it]
    df0[it] = df0[it] / volume_price

train_df = pd.merge(result_df_std[['ticker', 'clus']], df0, on = ['ticker'], how = 'inner')
train_df['return'] =  100*(train_df['smooth_price_1'] - train_df['smooth_price_10']) / train_df['smooth_price_10']

train_df['return'] =  100*(train_df['smooth_price_1'] - train_df['smooth_price_10']) / train_df['smooth_price_10']

for i in range(24, 0, -1):
    train_df[f'return_{i}'] = (train_df[f'smooth_price_{i}'] - train_df[f'smooth_price_{i+1}']) / train_df[f'smooth_price_{i+1}']

train_df['risk_return'] = train_df[return_columns[0:15]].std(axis=1)
train_df['risk_return_2'] = train_df['risk_return']**2


# create linear regression model: selecting a cluster
train_df_clus0 = train_df[train_df.clus == 1]
X = sm.add_constant(train_df_clus0[['risk_return', 'past_return_2_squared', 'risk_past_interaction']])
y = np.log(train_df_clus0['return'] + 1)  # Log transformation for stability
model = sm.OLS(y, X).fit()
print(model.summary())

# create logistic regression model: selecting a cluster
logist_model = LogisticRegression().fit(X_train, y_train)
pred_probs = logist_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred_probs)
ks = np.max(tpr - fpr)
print(f'AUC: {auc}, KS statistic: {ks}')


## apply SHAP to explain the model and key drivers
model = sm.OLS(y, X[cols]).fit()
# Print the summary of the model to get the p-values
explainer = shap.Explainer(model.predict, X[cols])
# Calculates the SHAP values
shap_values = explainer(X[cols])
shap.plots.bar(shap_values)

shap.plots.beeswarm(shap_values)

ORCL_index = X[X['ticker'] == 'ORCL'].index
ORCL_shap_values = shap_values[ORCL_index[0]]
shap.plots.bar(ORCL_shap_values)

##gether top key drivers for each stock into a DF
# Assuming alignment of 'tickers' with observations in 'X'
tickers = train_df_clus0['ticker'].values
# Prepare a dictionary to collect data
data = {
    'ticker': [],
    'top_feature1': [], 'top_feature2': [], 'top_feature3': [],
    'importance1': [], 'importance2': [], 'importance3': []}
# Iterate over SHAP values to extract top 3 influential features for each stock
for i, ticker in enumerate(tickers):
    sorted_indices = np.argsort(-np.abs(shap_values.values[i]))[:3]
    data['ticker'].append(ticker)
    data['top_feature1'].append(cols[sorted_indices[0]])
    data['top_feature2'].append(cols[sorted_indices[1]])
    data['top_feature3'].append(cols[sorted_indices[2]])
    data['importance1'].append(shap_values.values[i][sorted_indices[0]])
    data['importance2'].append(shap_values.values[i][sorted_indices[1]])
    data['importance3'].append(shap_values.values[i][sorted_indices[2]])
# Convert the collected data into a DataFrame for easy analysis
df_key_drivers = pd.DataFrame(data)


## Optimizing Stock Selection with Response Surface Methodology (RSM)

# Model fitting and response surface data generation
X = sm.add_constant(train_df_clus0[['risk_return', 'past_return_2_squared', 'risk_past_interaction']])
model = sm.OLS(y, X).fit()
x_range = np.linspace(DF['past_return'].min(), DF['past_return'].max(), 100)
y_range = np.linspace(DF['risk_return'].min(), DF['risk_return'].max(), 100)
x_grid, y_grid = np.meshgrid(x_range, y_range)
z_grid = model.params[0] + \
   model.params[1]*y_grid +  model.params[2]*x_grid**2 + \
   model.params[3]*x_grid*y_grid    

# Plotting the response surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Calculate distance to the optimal point
DF['distance'] = np.sqrt((DF['past_return'] -\
     optimal_x)**2 + (DF['risk_return'] - optimal_y)**2)
    
# Sort the DataFrame by the 'distance' column
DF_sorted = DF.sort_values('distance')
# Select the top 10 stocks with the smallest distance to the optimal point
top_10_stocks = DF_sorted.head(10)
print("Top 10 Stocks Closest to the Optimal Point:")
print(top_10_stocks[['ticker', 'past_return', 'risk_return', 'distance']])


# Find the optimal point for maximizing future return
optimal_idx = np.argmax(z_grid)
optimal_x = x_range[optimal_idx // 100]
optimal_y = y_range[optimal_idx % 100]
optimal_z = z_grid.flatten()[optimal_idx]
print(f"Optimal Point: past_return = {optimal_x}, risk_return = {optimal_y}, future return={optimal_z}")

