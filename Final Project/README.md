# Stock Price Prediction using LSTM

This project aims to predict future stock prices using a Long Short-Term Memory (LSTM) deep learning model. The system fetches historical stock data, preprocesses it, incorporates technical indicators as features, and trains an LSTM model to forecast the adjusted closing price for the next 15 days.



## 📜 Project Overview

The core of this project is a time-series forecasting model built with TensorFlow and Keras. It leverages the power of LSTMs, which are particularly well-suited for learning from sequential data like stock prices. By training on historical data, the model learns underlying patterns and trends to make future predictions.

---

## ✨ Features

-   **User-Driven Data Collection**: Fetches data for any stock ticker from Yahoo Finance using the `yfinance` library based on user-defined dates.
-   **Technical Indicator Integration**: Enhances model accuracy by using key technical indicators—**MACD** and **RSI**—as input features alongside the adjusted closing price.
-   **Data Visualization**: Provides clear visualizations of historical prices, technical indicators, model predictions vs. actual values, and the final 15-day forecast.
-   **LSTM Model**: Implements a stacked LSTM network with Dropout layers to prevent overfitting.
-   **Performance Evaluation**: Measures the model's accuracy using the **R² (R-squared) score**.

---

## 🛠️ Technologies & Libraries Used

-   **Programming Language**: Python 3
-   **Core Libraries**:
    -   `tensorflow`: For building and training the LSTM model.
    -   `yfinance`: For fetching historical stock market data.
    -   `pandas`: For data manipulation and analysis.
    -   `numpy`: For numerical operations.
    -   `scikit-learn`: For data scaling (`MinMaxScaler`) and model evaluation (`r2_score`).
    -   `matplotlib`: For creating and saving visualizations.

---

## 📈 Technical Indicators Explained

To provide the model with more context than just the price, two technical indicators were used:

1.  **MACD (Moving Average Convergence Divergence)**: This is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a stock's price. It helps identify changes in momentum, strength, and direction of a trend.
2.  **RSI (Relative Strength Index)**: This is a momentum oscillator that measures the speed and change of price movements. RSI values range from 0 to 100 and are typically used to identify overbought (above 70) or oversold (below 30) conditions in a stock.

---

## 🚀 How to Run the Project

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(To create the `requirements.txt` file, run `pip freeze > requirements.txt` after installing the libraries).*

3.  **Run the Script**:
    Execute the Python script from your terminal.
    ```bash
    python stock_predictor.py
    ```

4.  **Follow the Prompts**:
    -   The script will run interactively in your terminal. Enter the stock ticker, start date, and end date when prompted.
    -   As the script processes the data, visualization windows will pop up. **You must close each plot window to allow the script to continue to the next step.**

---

## 📊 Results and Visualizations

-   The final **R² score** and the 15-day price forecast will be printed directly to your terminal upon completion.
-   All visualizations generated during the run are automatically saved as **`.png` image files** in the project directory. This includes the indicator plots, the prediction vs. actual price comparison, and the final forecast chart.



---

## 📝 Author

-   [Your Name]
-   [Link to your GitHub profile]