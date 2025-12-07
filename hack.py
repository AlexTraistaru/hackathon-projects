import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from helper_function import validate_battery_actions

def parse_start(series: pd.Series) -> pd.Series:
    start_str = series.str.split(" - ").str[0]
    return pd.to_datetime(start_str, format="%d.%m.%Y %H:%M")

def optimize_day_dp(
    prices,
    capacity=10.0,
    initial_soc=5.0,
    step_unit=0.1,
    max_step=10,
):
    
    prices = np.asarray(prices, dtype=float)
    T = len(prices)

    # Folosim capacitatea completa (0 .. capacity in pasi de step_unit)
    eff_cap = capacity
    cap_steps = int(round(eff_cap / step_unit))          
    max_step_int = int(round(max_step / step_unit))     

    # Actiunile posibile in unitati de 0.1: [-max_step_int .. max_step_int]
    actions_int = np.arange(-max_step_int, max_step_int + 1, dtype=int)

    # dp[t, s] = profit maxim de la timpul t pana la final,
    dp = np.full((T + 1, cap_steps + 1), -1e18, dtype=float)
    next_action = np.zeros((T, cap_steps + 1), dtype=int)

    # La final (t = T), profitul viitor este 0 pentru orice SoC
    dp[T, :] = 0.0

    # Backward DP
    for t in range(T - 1, -1, -1):
        p = prices[t]
        for s in range(cap_steps + 1):
            best = -1e18
            best_a = 0
            for a_int in actions_int:
                s_next = s + a_int
                if s_next < 0 or s_next > cap_steps:
                    continue
                a = a_int * step_unit
                # profit = -a * p (cumperi => a>0 => cost mic; vinzi => a<0 => castig)
                reward = -a * p + dp[t + 1, s_next]
                if reward > best:
                    best = reward
                    best_a = a_int
            dp[t, s] = best
            next_action[t, s] = best_a

    # Reconstruim traiectoria optima plecand din SoC initial
    init_idx = int(round(initial_soc / step_unit))
    if init_idx > cap_steps:
        init_idx = cap_steps

    acts = np.zeros(T, dtype=float)
    s = init_idx
    for t in range(T):
        a_int = next_action[t, s]
        acts[t] = a_int * step_unit
        s = s + a_int
        if s < 0:
            s = 0
        if s > cap_steps:
            s = cap_steps

    return acts


def main():
    print("Pas 1/5: Citim setul de date...")
    sys.stdout.flush()

    df = pd.read_csv("Dataset.csv")
    sample = pd.read_csv("sample_submission.csv") 

    print("Pas 2/5: Generam label urile pentru regresie...")
    sys.stdout.flush()

    df["start"] = parse_start(df["Time interval (CET/CEST)"])
    sample["start"] = parse_start(sample["Time interval (CET/CEST)"])

    for frame in (df, sample):
        frame["hour"] = frame["start"].dt.hour
        frame["minute"] = frame["start"].dt.minute
        frame["quarter"] = frame["minute"] // 15
        frame["dayofweek"] = frame["start"].dt.dayofweek  # 0=luni
        frame["month"] = frame["start"].dt.month
        frame["dayofyear"] = frame["start"].dt.dayofyear
        frame["is_weekend"] = (frame["dayofweek"] >= 5).astype(int)
        frame["intraday_idx"] = frame["hour"] * 4 + frame["quarter"]

    df["date"] = df["start"].dt.date
    sample["date"] = sample["start"].dt.date

    print("Pas 3/5: Antrenam RandomForestRegressor pentru pret...")
    sys.stdout.flush()

    features = [
        "hour", #ora
        "quarter", #in ce sfert de ora ne aflam:1 2 3 sau 4
        "dayofweek", #0=luni ... 6=duminica
        "month", #luna
        "dayofyear", #ziua din an
        "is_weekend", #daca e weekend sau nu, 0 sau 1
        "intraday_idx", #indexul intervalului orar din zi: 0 - 95 (96 de intervale de 15 minute intr-o zi)
    ]

    X_train = df[features]
    y_train = df["Price"] #pretul de antrenare

    reg_price = RandomForestRegressor(
        n_estimators=1010, #numarul de arbori din padure
        max_depth=None,
        min_samples_leaf=10,
        max_features='log2',
        random_state=0,
        n_jobs=-1,
    )
    reg_price.fit(X_train, y_train)

    print("Pas 4/5: Prezicem preturi pe 8 zile...")
    sys.stdout.flush()

    X_sub = sample[features]
    sample["pred_price"] = reg_price.predict(X_sub)

    print("Pas 5/5: Calculez actiunile optime pe fiecare zi (algoritmul dynamic programming)...")
    sys.stdout.flush()

    actions = np.zeros(len(sample), dtype=float)

    unique_dates = sorted(sample["date"].unique())
    for d in unique_dates:
        idx = sample.index[sample["date"] == d]
        idx_sorted = idx.sort_values()
        prices_day = sample.loc[idx_sorted, "pred_price"].values

        day_actions = optimize_day_dp(
            prices_day,
            capacity=10.0,
            initial_soc=5.0,
            step_unit=0.1,
            max_step=10,   # putem sari cu pana la 10 MWh pe interval
        )

        actions[idx_sorted] = day_actions

    # Rotunjim explicit la o zecimala (multipli de 0.1)
    actions = np.round(actions, 1)

    print("Distributie actiuni pe cele 8 zile:")
    vals, counts = np.unique(actions, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"    {v:+.1f}: {c} ({c/len(actions)*100:.2f}%)")
    sys.stdout.flush()

    # Validam actiunile cu helper_function
    print("Validez actiunile cu validate_battery_actions...")
    sys.stdout.flush()

    is_valid, warnings = validate_battery_actions(
        actions.tolist(),
        capacity=10,
        initial_soc=5,
        timestep_hours=0.25,
        reset_daily=True,
        return_trace=False,
    )

    print(f"Rezultat validare: is_valid={is_valid}, nr_warnings={len(warnings)}") #afisam rezultatul validarii
    if len(warnings) > 0:
        print("  Primele cateva avertismente:")
        for w in warnings[:10]:
            print("   -", w)

    out = pd.DataFrame(
        {
            "Time interval (CET/CEST)": sample["Time interval (CET/CEST)"],
            "Position": actions,
        }
    )

    out.to_csv("incercaremaibunasper.csv", index=False)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
