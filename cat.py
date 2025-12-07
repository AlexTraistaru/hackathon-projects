import pandas as pd
import numpy as np
from cat import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# 1. Configurare

FILENAME = 'Dataset.csv'
BATTERY_CAPACITY = 10.0
DAILY_RESET_SOC = 5.0
FUTURE_DAYS = 8
FOLDS = 5  # Antrenăm 5 modele diferite și facem media acestora

# 2. Prelucrare de date

df_raw = pd.read_csv(FILENAME)
df_raw['dt'] = df_raw['Time interval (CET/CEST)'].str.split(' - ').str[0]
df_raw['dt'] = pd.to_datetime(df_raw['dt'], format='%d.%m.%Y %H:%M')
df_raw = df_raw.set_index('dt').sort_index()
df_raw = df_raw[['Price']].copy()

# 2.1 Interpolare pentru a nu pierde continuitatea

df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
df_raw['Price'] = df_raw['Price'].interpolate(method='time')
df_raw.dropna(inplace=True)

# 3. Feature Engineering

def create_features(df):

    df = df.copy()

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

   # Mapăm orele cu sin și cos pentru a evidenția faptul că orele 23 și 00 sunt apropiate

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

   # Ferestre de valori trecute

    df['lag_1'] = df['Price'].shift(1)

    df['lag_2'] = df['Price'].shift(2) # Viteza

    df['lag_96'] = df['Price'].shift(96) # Ieri

    df['lag_672'] = df['Price'].shift(672) # Săptămâna trecută

    # Media pe ultimele 24h

    df['roll_mean_24h'] = df['Price'].shift(1).rolling(window=96).mean()

    # Volatilitatea

    df['roll_std_24h'] = df['Price'].shift(1).rolling(window=96).std()

    # E prețul de acum mai mare ca media?

    df['ratio_mean'] = df['lag_1'] / (df['roll_mean_24h'] + 10)

    return df


df_features = create_features(df_raw).dropna()


FEATURES = ['hour', 'dayofweek', 'month', 'hour_sin', 'hour_cos', 
            'lag_1', 'lag_2', 'lag_96', 'lag_672', 
            'roll_mean_24h', 'roll_std_24h', 'ratio_mean']

TARGET = 'Price'

# 4. Antrenare

# TimeSeriesSplit taie datele în 5 felii progresive.
# Modelul 1 vede puțin, Modelul 5 vede tot.
# Asta simulează trecerea timpului și testează robustețea.

tscv = TimeSeriesSplit(n_splits=FOLDS)

models = []

X = df_features[FEATURES]
y = df_features[TARGET]

fold_idx = 1

for train_index, val_index in tscv.split(X):

    print(f"Se antrenează modelul {fold_idx}/{FOLDS}")

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Configurăm CatBoost să fie "Adânc" și Robust

    model = CatBoostRegressor(

        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=42 + fold_idx,
        verbose=False,
        early_stopping_rounds=100,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    models.append(model)

    # Check rapid
    score = mean_absolute_error(y_val, model.predict(X_val))

    #print(f"MAE pe fold-ul curent: {score:.2f}")

    fold_idx += 1

# 5. Predicție

future_steps = FUTURE_DAYS * 96
last_timestamp = df_raw.index[-1]
future_dates = [last_timestamp + pd.Timedelta(minutes=15 * (i+1)) for i in range(future_steps)]


df_history = df_features.copy()
df_future = pd.DataFrame(index=future_dates, columns=df_history.columns)
df_full = pd.concat([df_history, df_future])


df_full['hour'] = df_full.index.hour
df_full['dayofweek'] = df_full.index.dayofweek
df_full['month'] = df_full.index.month
df_full['hour_sin'] = np.sin(2 * np.pi * df_full['hour'] / 24)
df_full['hour_cos'] = np.cos(2 * np.pi * df_full['hour'] / 24)


start_idx = len(df_history)


for i in range(start_idx, len(df_full)):

    lag_1 = df_full.iloc[i-1]['Price']
    lag_2 = df_full.iloc[i-2]['Price']
    lag_96 = df_full.iloc[i-96]['Price']
    lag_672 = df_full.iloc[i-672]['Price'] if i >= 672 else lag_96


    rm_24h = df_full.iloc[i-96:i]['Price'].mean()
    rs_24h = df_full.iloc[i-96:i]['Price'].std()
    ratio_mean = lag_1 / (rm_24h + 10)


    # Input

    input_row = pd.DataFrame([[

        df_full.iloc[i]['hour'], df_full.iloc[i]['dayofweek'], df_full.iloc[i]['month'],
        df_full.iloc[i]['hour_sin'], df_full.iloc[i]['hour_cos'],

        lag_1, lag_2, lag_96, lag_672,
        rm_24h, rs_24h, ratio_mean

    ]], columns=FEATURES)

    

    # 2. Predicție de grup

    # Media celor 5 modele

    preds = [m.predict(input_row)[0] for m in models]
    avg_pred = np.mean(preds)

    
    # Salvăm
    df_full.iloc[i, df_full.columns.get_loc('Price')] = avg_pred


df_predicted = df_full.iloc[start_idx:].copy()

# 6. Acțiunile

actions = []
df_predicted['date'] = df_predicted.index.date
current_soc = DAILY_RESET_SOC

for date, day_data in df_predicted.groupby('date'):

    prices = day_data['Price'].values

    # Cumpărăm când e ieftin (Top 20%)
    # Vindem când e scump (Top 20%)

    low_thresh = np.percentile(prices, 20)
    high_thresh = np.percentile(prices, 80)

    current_soc = DAILY_RESET_SOC

    for k in range(len(prices)):

        price = prices[k]
        target_soc = current_soc

        # Prețul negativ este prioritar

        if price < 0:
            target_soc = BATTERY_CAPACITY

        # Zonă Sigură de Cumpărare

        elif price <= low_thresh:
            target_soc = BATTERY_CAPACITY

            

        # Zonă Sigură de Vânzare

        elif price >= high_thresh:
            target_soc = 0

        # Golire la final

        if k == len(prices) - 1:
            target_soc = 0

        action = target_soc - current_soc

        
        # Filtru zgomot
        if abs(action) < 0.1: action = 0
            

        current_soc += action
        current_soc = max(0.0, min(BATTERY_CAPACITY, current_soc))
        actions.append(action)

# 7. Export

submission = pd.DataFrame({
    'Time interval (CET/CEST)': df_predicted.index,
    'Position': actions
})

submission['Time interval (CET/CEST)'] = submission['Time interval (CET/CEST)'].apply(
    lambda x: f"{x.strftime('%d.%m.%Y %H:%M')} - {(x + pd.Timedelta(minutes=15)).strftime('%d.%m.%Y %H:%M')}"
)

filename = 'catboost.csv'

submission.to_csv(filename, index=False)