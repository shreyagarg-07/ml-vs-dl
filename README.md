# ml-vs-dl
Google (GOOG) Stock Price Prediction — ML vs Deep Learning
A comprehensive comparison of 6 machine learning and deep learning models for predicting Google stock prices using real historical data from Yahoo Finance (January 2019 – December 2023).

🧠 Models Implemented
#ModelType1Linear RegressionClassical ML2Support Vector Regression (SVR – RBF kernel)Classical ML3Random ForestEnsemble ML4Gradient BoostingEnsemble ML5LSTM-equivalent Deep MLPDeep Learning6GRU-equivalent Deep MLPDeep Learning

Note: LSTM and GRU are approximated using deep MLPs trained on 30-day sliding windows of closing prices — no TensorFlow or PyTorch required.


📊 Dataset

Source: Yahoo Finance historical OHLCV data
Ticker: GOOG (Alphabet Inc.)
Period: January 2019 – December 2023
Size: 1,192 trading days (after cleaning)
File required: GOOG_2019_2023_with_features.csv

Features Used (16 technical indicators)
Open, High, Low, Volume, MA_10, MA_50, EMA_12, MACD, RSI, BB_Width, Volatility, Vol_MA5, Lag_1, Lag_2, Lag_3, Lag_5

⚙️ Methodology

Train/Test Split: 80% training (~953 days), 20% testing (~239 days) — chronological, never shuffled
Scaling: Min-Max normalization [0, 1] fitted only on training data to prevent data leakage
Sequence length (DL models): 30-day sliding window of closing prices


📐 Evaluation Metrics
MetricDescriptionRMSERoot Mean Squared Error — penalizes large errorsMAEMean Absolute Error — average absolute deviationR²Coefficient of Determination — variance explainedMAPEMean Absolute Percentage Error — scale-independentDirectional Accuracy% of days where up/down movement is correctly predicted

🏆 Key Results
AwardModelScore🥇 Best RMSELinear Regression$0.60🥇 Best Directional AccuracySVR (RBF)89.08%🥇 Overall WinnerML models outperformed Deep Learning
Why did ML models win?

ML models had access to 16 engineered features vs. only 30-day price sequences for DL
The dataset (~953 training samples) is relatively small for deep learning
The 2023 test window was relatively smooth, favouring trend-following features


📁 Project Structure
📦 GOOG-Stock-Prediction
 ┣ 📓 GOOG_Stock_Prediction_Real.ipynb   # Main notebook
 ┣ 📄 GOOG_2019_2023_with_features.csv   # Dataset (see setup below)
 ┗ 📄 README.md

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/your-username/GOOG-Stock-Prediction.git
cd GOOG-Stock-Prediction
2. Install dependencies
bashpip install numpy pandas matplotlib scikit-learn
3. Get the dataset
Option A — Use the provided CSV (if included in the repo): place GOOG_2019_2023_with_features.csv in the project root.
Option B — Download fresh data with yfinance:
pythonimport yfinance as yf
df = yf.download('GOOG', start='2019-01-01', end='2023-12-31')
df.to_csv('GOOG_raw.csv')
Then engineer the 16 technical indicator features as shown in Section 1 of the notebook.
4. Run the notebook
Open GOOG_Stock_Prediction_Real.ipynb in Jupyter or Google Colab and run all cells top to bottom.

📉 Visualizations Included

GOOG price history (2019–2023)
Actual vs. Predicted prices for all 6 models
Side-by-side bar charts comparing RMSE, MAE, and Directional Accuracy across all models


💡 Practical Takeaway
For trading signals, SVR's 89% directional accuracy is the most valuable metric — it correctly predicted whether Google stock would rise or fall on nearly 9 out of 10 test days.

🛠️ Requirements

Python 3.7+
numpy
pandas
matplotlib
scikit-learn
