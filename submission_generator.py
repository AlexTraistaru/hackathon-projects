import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
import matplotlib.pyplot as plt

# Ignorăm warning-urile pentru a păstra consola curată
warnings.filterwarnings('ignore')

FILENAME = 'Dataset.csv'
BATTERY_CAPACITY = 10.0
DAILY_RESET_SOC = 5.0
VALIDATION_DAYS = 14  # Ultimele 2 săptămâni pentru testare
FUTURE_DAYS = 8       # Câte zile prezicem

def create_features(df):
    """
    Creează variabilele explicative pentru model.
    """
    df = df.copy()
    
    # --- A. Timpul Ciclic (Ceasul Matematic) ---
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # Sinus/Cosinus pentru a modela ciclicitatea (ora 23 e lângă ora 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # --- B. Lag-uri (Istoricul Prețului) ---
    # Lag 1: Prețul imediat anterior
    df['lag_1'] = df['Price'].shift(1)
    # Lag 96: Prețul de ieri la aceeași oră (Foarte important!)
    df['lag_96'] = df['Price'].shift(96)
    # Lag 672: Prețul de săptămâna trecută
    df['lag_672'] = df['Price'].shift(672)
    
    # --- C. Statistici Mobile (Trenduri) ---
    # Media prețului pe ultimele 24 de ore (96 intervale)
    # Ne ajută să vedem dacă suntem într-o perioadă scumpă sau ieftină
    df['rolling_mean_24h'] = df['Price'].shift(1).rolling(window=96).mean()
    
    return df

print(">>> [1/5] Incarc si procesez datele...")

# Citire CSV
df_raw = pd.read_csv(FILENAME)

# Curățare format dată ("01.02.2021 00:00 - ..." -> datetime)
df_raw['dt'] = df_raw['Time interval (CET/CEST)'].str.split(' - ').str[0]
df_raw['dt'] = pd.to_datetime(df_raw['dt'], format='%d.%m.%Y %H:%M')
df_raw = df_raw.set_index('dt').sort_index()

# Păstrăm doar coloana Price
df_raw = df_raw[['Price']].copy()

# Generăm feature-urile pe tot istoricul disponibil
df_features = create_features(df_raw)


print(">>> [2/5] Antrenez modelul (Etapa de Validare)...")

# Definim coloanele de intrare și ținta
FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
            'lag_1', 'lag_96', 'lag_672', 'rolling_mean_24h']
TARGET = 'Price'

# Împărțim în Train și Validation (ultimele 14 zile)
split_idx = len(df_features) - (VALIDATION_DAYS * 96)
train_df = df_features.iloc[:split_idx].dropna()
val_df = df_features.iloc[split_idx:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_val = val_df[FEATURES]
y_val = val_df[TARGET]

# Modelul de test (cu Early Stopping)
model_val = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:pseudohubererror', # Robust la spike-uri
    n_jobs=-1,
    early_stopping_rounds=50,         # Se oprește dacă nu mai învață
    random_state=42
)

model_val.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# Calculăm eroarea doar ca să știm cum stăm
val_preds = model_val.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print(f"    Eroare (MAE) pe validare: {mae:.2f}")
print("    (Notă: E normal să fie mare dacă prețurile recente au explodat. Re-antrenarea va rezolva asta.)")


print(">>> [3/5] Re-antrenez modelul pe TOT istoricul (fără early stopping)...")

full_data = df_features.dropna()

# Modelul FINAL - Fără early_stopping_rounds în constructor!
final_model = xgb.XGBRegressor(
    n_estimators=1500,       # Număr fix de iteratii, suficient de mare
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='reg:pseudohubererror',
    n_jobs=-1,
    random_state=42
    # NU punem early_stopping_rounds aici
)

final_model.fit(full_data[FEATURES], full_data[TARGET], verbose=False)
print("    Model final antrenat cu succes.")

print(">>> [4/5] Generez predictiile pentru următoarele 8 zile...")

future_steps = FUTURE_DAYS * 96
last_timestamp = df_raw.index[-1]
future_dates = [last_timestamp + pd.Timedelta(minutes=15 * (i+1)) for i in range(future_steps)]

# Creăm un DataFrame mare care conține Istoricul + Viitorul (gol)
df_history = df_features.copy()
df_future_empty = pd.DataFrame(index=future_dates, columns=df_history.columns)
df_full = pd.concat([df_history, df_future_empty])

# Pre-calculăm feature-urile de timp pentru viitor
df_full['hour'] = df_full.index.hour
df_full['dayofweek'] = df_full.index.dayofweek
df_full['hour_sin'] = np.sin(2 * np.pi * df_full['hour'] / 24)
df_full['hour_cos'] = np.cos(2 * np.pi * df_full['hour'] / 24)
df_full['dow_sin'] = np.sin(2 * np.pi * df_full['dayofweek'] / 7)
df_full['dow_cos'] = np.cos(2 * np.pi * df_full['dayofweek'] / 7)

# Bucla Recursivă
start_idx = len(df_history)

for i in range(start_idx, len(df_full)):
    # Luăm valorile necesare din trecut (folosind .iloc)
    
    # Lag 1: Predicția anterioară (sau realitatea dacă e primul pas)
    lag_1 = df_full.iloc[i-1]['Price']
    
    # Lag 96: Ieri
    lag_96 = df_full.iloc[i-96]['Price']
    
    # Lag 672: Săptămâna trecută (Fallback la Lag 96 dacă nu avem destul istoric)
    lag_672 = df_full.iloc[i-672]['Price'] if i >= 672 else lag_96
    
    # Rolling Mean: Media ultimelor 96 de valori
    rm_24h = df_full.iloc[i-96:i]['Price'].mean()
    
    # Construim rândul pentru predicție
    # Ordinea coloanelor trebuie să fie IDENTICĂ cu cea de la antrenare (FEATURES)
    input_features = [
        df_full.iloc[i]['hour_sin'], 
        df_full.iloc[i]['hour_cos'],
        df_full.iloc[i]['dow_sin'], 
        df_full.iloc[i]['dow_cos'],
        lag_1, 
        lag_96, 
        lag_672, 
        rm_24h
    ]
    
    # Prezicem
    # Reshape pentru că XGBoost așteaptă matrice 2D
    pred_val = final_model.predict([input_features])[0]
    
    # Salvăm predicția în DataFrame
    df_full.iloc[i, df_full.columns.get_loc('Price')] = pred_val

# Extragem doar partea prezisă
df_predicted = df_full.iloc[start_idx:].copy()


print(">>> [5/5] Calculez acțiunile (Strategia Zone Dinamice)...")

actions = []
df_predicted['date'] = df_predicted.index.date
current_soc = DAILY_RESET_SOC

# Procesăm zi cu zi
for date, day_data in df_predicted.groupby('date'):
    prices = day_data['Price'].values
    
    # Pentru a minimiza acțiunile de 0, îngustăm zona neutră.
    # Cumpărăm în cele mai ieftine 35% momente
    # Vindem în cele mai scumpe 35% momente
    # Zona neutră rămâne doar 30% din timp
    low_thresh = np.percentile(prices, 35)
    high_thresh = np.percentile(prices, 65)
    
    current_soc = DAILY_RESET_SOC # Reset la 5 MWh la startul zilei
    
    for k in range(len(prices)):
        price = prices[k]
        target_soc = current_soc # Default: Stai pe loc
        
        # PRIORITATE 1: Bani Gratis (Preț Negativ)
        if price < 0:
            target_soc = BATTERY_CAPACITY
            
        # PRIORITATE 2: Ieftin -> Încarcă
        elif price <= low_thresh:
            target_soc = BATTERY_CAPACITY
            
        # PRIORITATE 3: Scump -> Descarcă
        elif price >= high_thresh:
            target_soc = 0
            
        # PRIORITATE 4: Final de zi -> Golește tot (fără penalizare)
        if k == len(prices) - 1:
            target_soc = 0
            
        # Calculăm acțiunea
        action = target_soc - current_soc
        
        # Filtru de zgomot (să nu facem tranzacții minuscule sub 0.1 MWh)
        if abs(action) < 0.1:
            action = 0
            
        # Aplicăm acțiunea virtual
        current_soc += action
        
        # Corecție limite (float precision)
        current_soc = max(0.0, min(BATTERY_CAPACITY, current_soc))
        
        actions.append(action)

submission = pd.DataFrame({
    'Time interval (CET/CEST)': df_predicted.index,
    'Position': actions
})

# Formatare String (Importantă pentru checker!)
submission['Time interval (CET/CEST)'] = submission['Time interval (CET/CEST)'].apply(
    lambda x: f"{x.strftime('%d.%m.%Y %H:%M')} - {(x + pd.Timedelta(minutes=15)).strftime('%d.%m.%Y %H:%M')}"
)

submission.to_csv('submission_final.csv', index=False)
print(">>> SUCCES! Fișier generat: submission_final.csv")
print(submission.head())