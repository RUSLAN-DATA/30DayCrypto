import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# Инициализация клиента ccxt для получения данных
exchange = ccxt.binance()

# Список популярных криптовалют
popular_cryptos = {
    1: 'BTC/USDT',  # Bitcoin
    2: 'ETH/USDT',  # Ethereum
    3: 'BNB/USDT',  # Binance Coin
    4: 'ADA/USDT',  # Cardano
    5: 'XRP/USDT',  # Ripple
    6: 'SOL/USDT',  # Solana
    7: 'DOGE/USDT',  # Dogecoin
    8: 'DOT/USDT',  # Polkadot
    9: 'LTC/USDT',  # Litecoin
    10: 'MATIC/USDT'  # Polygon
}

# Функция для получения исторических данных (примерно два года, по 730 дней)
def fetch_ohlcv(symbol='BTC/USDT', timeframe='1d', limit=730):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Функция для подготовки данных для LSTM
def prepare_data(df, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(time_step, len(df_scaled)):
        X_train.append(df_scaled[i-time_step:i, 0])
        y_train.append(df_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

# Функция для создания и обучения модели LSTM
def create_and_train_lstm(X_train, y_train):
    model = Sequential()
    
    # Добавляем явный слой ввода
    model.add(Input(shape=(X_train.shape[1], 1)))
    
    # LSTM слои
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=20)

    return model

# Функция для предсказания цен
def predict_future_prices(model, df, scaler, time_step=60, future_days=30):
    last_days = df['close'].values[-time_step:]
    last_days_scaled = scaler.transform(last_days.reshape(-1, 1))

    X_test = []
    X_test.append(last_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_prices_scaled = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

    future_prices = [predicted_prices[0][0]]
    for _ in range(future_days - 1):
        new_sequence = np.concatenate([last_days_scaled[1:], predicted_prices_scaled], axis=0)
        new_sequence = np.reshape(new_sequence, (1, time_step, 1))
        predicted_prices_scaled = model.predict(new_sequence)
        future_prices.append(scaler.inverse_transform(predicted_prices_scaled)[0][0])
        last_days_scaled = new_sequence[0]

    return future_prices

# Функция для визуализации данных
def plot_results(df, future_prices, crypto_name):
    # Создаем график для отображения реальных и предсказанных цен
    plt.figure(figsize=(10, 6))

    # Реальные данные
    plt.plot(df['timestamp'], df['close'], label="Historical Prices")

    # Прогнозируемые цены
    future_dates = pd.date_range(df['timestamp'].iloc[-1], periods=len(future_prices) + 1, freq='D')[1:]
    plt.plot(future_dates, future_prices, label="Predicted Prices", linestyle='--')

    plt.title(f"{crypto_name} Price Prediction for Next 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Price (USDT)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Основная функция
def main():
    print("Select a cryptocurrency to analyze:")
    for key, value in popular_cryptos.items():
        print(f"{key}: {value.split('/')[0]}")

    choice = int(input("Enter the number of the cryptocurrency: "))
    
    if choice not in popular_cryptos:
        print("Invalid choice. Exiting.")
        return

    symbol = popular_cryptos[choice]
    crypto_name = symbol.split('/')[0]
    
    df = fetch_ohlcv(symbol)

    # Подготовка данных для обучения
    time_step = 60
    X_train, y_train, scaler = prepare_data(df, time_step)
    
    # Создание и обучение модели
    model = create_and_train_lstm(X_train, y_train)
    
    # Прогноз на будущее (30 дней)
    future_days = 30
    future_prices = predict_future_prices(model, df, scaler, time_step, future_days)
    
    # Визуализация результатов
    plot_results(df, future_prices, crypto_name)

if __name__ == "__main__":
    main()
