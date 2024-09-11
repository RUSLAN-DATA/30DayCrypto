# 30DayCrypto
# CryptoLSTM

A tool for predicting the prices of popular cryptocurrencies for the next 30 days using an LSTM neural network. The project allows you to select from 10 popular cryptocurrencies and visualize future prices based on the last 2 years of historical data.

## Features

- Predicts prices for the next 30 days using LSTM (Long Short-Term Memory) neural networks.
- Supports 10 popular cryptocurrencies (BTC, ETH, BNB, ADA, XRP, SOL, DOGE, DOT, LTC, MATIC).
- Visualizes predicted prices along with historical data.

## Requirements

- Python 3.8 or higher
- The following Python libraries:
  - `ccxt`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow`

## Installation and Usage

1. Clone the repository or download the project to your machine.
2. Install the required libraries:
    ```bash
    pip install ccxt pandas numpy matplotlib scikit-learn tensorflow
    ```
3. Run the script:
    ```bash
    python crypto_predictor.py
    ```
4. Choose the cryptocurrency to analyze and predict.
5. After running the script, a graph with the price predictions for the next 30 days will be displayed.

## Cryptocurrency Selection

After starting the script, you will be prompted to select one of the 10 popular cryptocurrencies using a number input:

1. **BTC** - Bitcoin
2. **ETH** - Ethereum
3. **BNB** - Binance Coin
4. **ADA** - Cardano
5. **XRP** - Ripple
6. **SOL** - Solana
7. **DOGE** - Dogecoin
8. **DOT** - Polkadot
9. **LTC** - Litecoin
10. **MATIC** - Polygon

## Visualization

Once the model finishes running, it will predict the prices for the next 30 days, and the result will be visualized in a graph that opens automatically.
