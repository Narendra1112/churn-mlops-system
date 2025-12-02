import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telecom_processed.csv")

REPORT_DIR = os.path.join(BASE_DIR, "monitoring")
REPORT_JSON = os.path.join(REPORT_DIR, "drift_report.json")
REPORT_HTML = os.path.join(REPORT_DIR, "drift_report.html")

# 1. Load baseline training data

def load_baseline_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Baseline data missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df

# 2. SIMULATE new batch data (manual drift)

def generate_synthetic_batch(baseline_df, drift_scale=0.35):
    df_new = baseline_df.copy()

    # Numeric drift
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df_new[col] = df_new[col] * (1 + np.random.normal(0, drift_scale, df_new.shape[0]))

    # Categorical drift
    cat_cols = df_new.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        df_new[col] = np.random.choice(df_new[col].unique(), size=len(df_new))

    return df_new



# 3. KS test (numeric)

def ks_test_numeric(baseline_df, new_df):
    results = {}

    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        stat, p = ks_2samp(baseline_df[col], new_df[col])
        results[col] = {
            "p_value": float(p),
            "drift": bool(p < 0.05)
        }

    return results



# 4. Jensen–Shannon divergence (categorical)

def js_divergence(p, q):
    p = np.array(p, dtype=float) + 1e-8
    q = np.array(q, dtype=float) + 1e-8

    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def categorical_drift(baseline_df, new_df):
    results = {}

    cat_cols = baseline_df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in cat_cols:
        base_dist = baseline_df[col].value_counts(normalize=True)
        new_dist = new_df[col].value_counts(normalize=True)

        # Align indexes
        all_idx = sorted(set(base_dist.index) | set(new_dist.index))
        p = base_dist.reindex(all_idx, fill_value=0)
        q = new_dist.reindex(all_idx, fill_value=0)

        js = js_divergence(p, q)

        results[col] = {
            "js_divergence": float(js),
            "drift": bool(js > 0.15)
        }

    return results


# 5. JSON-safe conversion (NO numpy.bool8)

def convert_numpy(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.int32, np.int64, int)):
        return int(value)
    if isinstance(value, (np.float32, np.float64, float)):
        return float(value)
    return value


# 6. Save JSON + HTML reports

def save_reports(numeric_res, categorical_res):
    os.makedirs(REPORT_DIR, exist_ok=True)

    numeric_clean = {
        col: {k: convert_numpy(v) for k, v in res.items()}
        for col, res in numeric_res.items()
    }

    categorical_clean = {
        col: {k: convert_numpy(v) for k, v in res.items()}
        for col, res in categorical_res.items()
    }

    final_report = {
        "numeric_drift": numeric_clean,
        "categorical_drift": categorical_clean
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(final_report, f, indent=4)

    with open(REPORT_HTML, "w") as f:
        f.write("<h1>Data Drift Report</h1>")

        f.write("<h2>Numeric Columns</h2><ul>")
        for col, res in numeric_clean.items():
            f.write(f"<li><b>{col}</b>: p={res['p_value']:.4f}, Drift={res['drift']}</li>")
        f.write("</ul>")

        f.write("<h2>Categorical Columns</h2><ul>")
        for col, res in categorical_clean.items():
            f.write(f"<li><b>{col}</b>: JS={res['js_divergence']:.4f}, Drift={res['drift']}</li>")
        f.write("</ul>")

    print(f" Drift report saved at:\n{REPORT_JSON}\n{REPORT_HTML}")


# 7. MAIN

def run_drift_monitor():
    print("Running manual data drift monitor...")

    baseline_df = load_baseline_data()
    new_batch_df = generate_synthetic_batch(baseline_df)

    numeric_res = ks_test_numeric(baseline_df, new_batch_df)
    categorical_res = categorical_drift(baseline_df, new_batch_df)

    save_reports(numeric_res, categorical_res)

    print(" Drift analysis completed.")


if __name__ == "__main__":
    run_drift_monitor()
