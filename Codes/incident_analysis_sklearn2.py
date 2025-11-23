"""
incident_analysis.py

Option A with scikit-learn and plots.

Changes from previous version:
1. Asks the user to supply the path to the Incident Excel dataset.
2. Generates two plots:
   - Yearly incidents with fitted linear regression trend.
   - Bar chart of top N states by incident count.
3. Asks the user for an output folder where CSV files and plots will be saved.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ================== CONFIGURATION ==================

# If your file has only one sheet, keep this as None
SHEET_NAME = None          # or "Incident" if you know the sheet name

# >>> make sure these match your columns <<<
INCIDENT_ID_COLUMN = "Incident_ID"
DATE_COLUMN        = "Date"
STATE_COLUMN       = "State"

INSPECT_ONLY = False   # set to True if you want to re-check columns


# ================== HELPER FUNCTIONS ==================

def load_incident_data(path: str, sheet_name=None) -> pd.DataFrame:
    """Load the Incident data from an Excel file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if sheet_name is None:
        df = pd.read_excel(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    return df


def prepare_incident_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, extract Year, and ensure required columns exist."""
    if DATE_COLUMN not in df.columns:
        raise KeyError(
            f"Expected date column '{DATE_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if STATE_COLUMN not in df.columns:
        raise KeyError(
            f"Expected state column '{STATE_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if INCIDENT_ID_COLUMN not in df.columns:
        # auto-detect if needed
        for col in df.columns:
            cl = col.lower()
            if "incident" in cl and "id" in cl:
                print(f"Auto-detected incident ID column as '{col}'.")
                df = df.rename(columns={col: INCIDENT_ID_COLUMN})
                break

    if INCIDENT_ID_COLUMN not in df.columns:
        raise KeyError(
            f"Expected incident id column '{INCIDENT_ID_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # parse dates and extract Year
    df["Date_parsed"] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["Year"] = df["Date_parsed"].dt.year

    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    return df


def analyze_temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q1: yearly incident counts.
    """
    yearly = (
        df.groupby("Year")
        .agg(incidents=(INCIDENT_ID_COLUMN, "count"))
        .reset_index()
        .sort_values("Year")
    )
    return yearly


def fit_time_trend_model(yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a LinearRegression model: incidents ~ Year.

    Returns the same DataFrame with an extra column 'predicted_incidents'
    and prints model parameters + R^2.
    """
    X = yearly[["Year"]].values          # shape (n_samples, 1)
    y = yearly["incidents"].values       # shape (n_samples,)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    yearly = yearly.copy()
    yearly["predicted_incidents"] = y_pred

    r2 = r2_score(y, y_pred)

    print("\n=== scikit-learn Linear Regression (incidents ~ Year) ===")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"Slope:     {model.coef_[0]:.3f} incidents per year")
    print(f"R^2:       {r2:.3f}")

    return yearly


def analyze_state_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q2: incidents per state.
    """
    state_counts = (
        df.groupby(STATE_COLUMN)
        .agg(incidents=(INCIDENT_ID_COLUMN, "count"))
        .reset_index()
        .sort_values("incidents", ascending=False)
    )
    return state_counts


def plot_yearly_trend(yearly_with_model: pd.DataFrame, output_dir: str, filename: str = "yearly_trend.png") -> None:
    """
    Plot actual incidents per year and fitted linear trend.
    Save the figure to output_dir/filename.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_with_model["Year"], yearly_with_model["incidents"],
             marker="o", linestyle="-", label="Actual incidents")
    plt.plot(yearly_with_model["Year"], yearly_with_model["predicted_incidents"],
             linestyle="--", label="Fitted linear trend")

    plt.title("School Gunfire Incidents per Year (1966–2025)")
    plt.xlabel("Year")
    plt.ylabel("Number of Incidents")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved yearly trend plot to: {out_path}")


def plot_state_topN(state_counts: pd.DataFrame, output_dir: str, top_n: int = 15,
                    filename: str = "state_top_incidents.png") -> None:
    """
    Plot horizontal bar chart of top N states by incident count.
    Save the figure to output_dir/filename.
    """
    top_states = state_counts.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_states["State"], top_states["incidents"])
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} States by Number of School Gunfire Incidents")
    plt.xlabel("Number of Incidents")
    plt.ylabel("State")
    plt.grid(axis="x")

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved state-level plot to: {out_path}")


# ================== MAIN ==================

def main():
    # --- 1. Ask user for input Excel path ---
    excel_path = input("Enter full path to the Incident Excel file: ").strip().strip('"')
    if not excel_path:
        print("No path provided. Exiting.")
        return

    # --- 2. Ask user for output directory ---
    output_dir = input("Enter output folder path for CSVs and plots: ").strip().strip('"')
    if not output_dir:
        print("No output folder provided. Exiting.")
        return

    if not os.path.isdir(output_dir):
        print(f"Output folder does not exist: {output_dir}")
        return

    print(f"\nLoading Incident data from: {excel_path}")
    incident_df = load_incident_data(excel_path, sheet_name=SHEET_NAME)
    print(f"Loaded Incident data with shape: {incident_df.shape}")

    if INSPECT_ONLY:
        print("\nINSPECT_ONLY is True. Available columns in Incident sheet:")
        print(list(incident_df.columns))
        print(
            "\nEdit INCIDENT_ID_COLUMN, DATE_COLUMN, and STATE_COLUMN if needed, "
            "then set INSPECT_ONLY = False and run again."
        )
        return

    print("\nPreparing data (parsing dates, extracting Year)...")
    incident_df = prepare_incident_data(incident_df)
    print(f"Data after preparation: {incident_df.shape}")

    # -------- Q1: Temporal Trends + Model --------
    print("\n=== Q1: Temporal Trends (Yearly) ===")
    yearly = analyze_temporal_trends(incident_df)

    # just to print all the rows in Python
    pd.set_option("display.max_rows", None)
    print(yearly)

    # Fit scikit-learn Linear Regression
    yearly_with_model = fit_time_trend_model(yearly)

    yearly_out = os.path.join(output_dir, "yearly_incidents_sklearn.csv")
    yearly_with_model.to_csv(yearly_out, index=False)
    print(f"Saved yearly trends + model predictions to: {yearly_out}")

    # Plot yearly trend
    plot_yearly_trend(yearly_with_model, output_dir)

    # -------- Q2: State-Level Patterns --------
    print("\n=== Q2: State-Level Patterns ===")
    state_counts = analyze_state_patterns(incident_df)

    # just to print all the rows in Python
    pd.set_option("display.max_rows", None)
    print(state_counts)

    state_out = os.path.join(output_dir, "state_incidents_sklearn.csv")
    state_counts.to_csv(state_out, index=False)
    print(f"Saved state-level counts to: {state_out}")

    # Plot top states
    plot_state_topN(state_counts, output_dir, top_n=15)

    print("\nDone.")


if __name__ == "__main__":
    main()
