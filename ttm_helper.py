import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import list_repo_refs

from tsfm_public import (
    # TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
    # TimeSeriesPreprocessor,
    # TinyTimeMixerConfig,
    # TrackingCallback,
    # count_parameters,
    # get_datasets,
)

task_configs = {
    "OIL" : ("OIL", "oil.csv", "spread", True, True),
    "REGIME" : ("REGIME", "regime_regression.csv", "risk_on_score", True, False),
    "SP500_VOL" : ("SP500_VOL", "regime_regression.csv", "RV_target", False, False),
    "PCA_TREASURY" : ("PCA_TREASURY", "treasury_yields.csv", "PC1", True, True),
    "10y_FORECAST" : ("10y_FORECAST", "10y_yield_forecast.csv", "10y", True, False),
    "10y_FORECAST_DIFF" : ("10y_FORECAST", "10y_yield_forecast_diff.csv", "10y", True, False),
    "FX_VOL" : ("FX_VOL", "fx_vol_forecasting.csv", "realized_vol", True, False),
    "SPREAD" : ("SPREAD", "equity_spread_forecasting.csv", "spread", True, False),
}

column_rename_dict = {
    '1 Mo': '1M Treasury yield',
    '2y': '2Y Treasury yield',
    '10y': '10Y Treasury yield',
    '5y-2y': '5Y–2Y slope',
    'RGDP_SA_P1Q1QL1AR': 'Real GDP growth (QoQ, SAAR)',
    'INTRGDP_NSA_P1M1ML12_3MMA': 'Intuitive GDP growth (% YoY, 3MMA)',
    'EMPL_NSA_P1M1ML12_3MMA': 'Employment growth (% YoY, 3MMA)',
    'UNEMPLRATE_SA_3MMA': 'Unemployment rate (SA, 3MMA)',
    'M2': 'Money supply (M2)',
    'VIX': 'VIX',
    'MOVE': 'MOVE',
    'IMPINFM1Y_NSA': 'Market-implied 1Y inflation expectation',
    'IMPINFM5Y_NSA': 'Market-implied 5Y inflation expectation',
    'equity_returns': 'Equity returns',
    'diff_ma_5': 'EMA-based volatility (5-day half-life)',
    'BBP_5_2.0': 'Bollinger band position',
    'bbb_z': 'Short-term trend',
    'volatility': 'Volatility (technical)'
}

def calculate_21_day_ret(df):
    df = df.copy()
    df.target_prediction = df.target_prediction.apply(lambda x:x[:21])
    df.target = df.target.apply(lambda x:x[:21])
    # Convert percentage changes to cumulative returns
    df['target_prediction'] = df['target_prediction'].apply(lambda x: np.cumprod(1 + np.array(x))[-1] - 1)
    df['target'] = df['target'].apply(lambda x: np.cumprod(1 + np.array(x))[-1] - 1)
    df = df.set_index('date')
    return df

def calculate_21_day_ret_mse(df):
    """
    Calculate MSE of model predictions vs naive model (variance). To be used only for 21 
    
    Args:
        df: DataFrame with target and target_prediction columns containing arrays
        
    Returns:
        tuple: (model_mse, naive_mse)
    """
    # Make copy to avoid modifying original
    df = calculate_21_day_ret(df.copy())
    model_mse = ((df.target-df.target_prediction)**2).mean()
    naive_mse = df.target.var()
    
    return model_mse, naive_mse


def plot_forecast_for_date(df, date, max_horizon=30):
    """
    Plots predicted vs actual values for a given date up to max_horizon.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'date', 'target_prediction', and 'target' columns.
        date (str or pd.Timestamp): Date to plot.
        max_horizon (int): Max forecast horizon to display (default=30).
    """
    row = df[df["date"] == date]
    if row.empty:
        print(f"No entry found for date: {date}")
        return
    
    preds = np.array(row["target_prediction"].values[0])[:max_horizon]
    trues = np.array(row["target"].values[0])[:max_horizon]
    horizons = np.arange(1, len(preds) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(horizons, trues, marker='o', label="Actual")
    plt.plot(horizons, preds, marker='x', label="Predicted")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Target Value")
    plt.title(f"Forecast vs Actual on {date}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_model_mse(df, max_horizon=21):
    """
    Calculate MSE for model predictions across different horizons.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'target' and 'target_prediction' columns
        max_horizon (int): Maximum forecast horizon to include
        
    Returns:
        np.ndarray: Array of MSE values for each horizon
    """
    # Ensure values are numpy arrays
    df = df.copy()
    df["target"] = df["target"].apply(np.array)
    df["target_prediction"] = df["target_prediction"].apply(np.array)
    
    # Stack into matrices
    true_matrix = np.vstack(df["target"].values)
    pred_matrix = np.vstack(df["target_prediction"].values)
    
    return ((pred_matrix - true_matrix) ** 2).mean(axis=0)[:max_horizon]

def calculate_naive_mse(df, is_target_diff = True, max_horizon=21):
    """
    Calculate MSE for naive baseline predictions across different horizons.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'target' column
        is_target_diff (bool): Whether target is a difference
        max_horizon (int): Maximum forecast horizon to include
        
    Returns:
        np.ndarray: Array of MSE values for each horizon
    """
    # Ensure values are numpy arrays
    df = df.copy()
    df["target"] = df["target"].apply(np.array)
    
    # Stack into matrix
    true_matrix = np.vstack(df["target"].values)
    
    if is_target_diff:
        naive_preds = np.zeros_like(true_matrix)
    else:
        naive_preds = np.repeat(true_matrix[:, [0]], true_matrix.shape[1], axis=1)
    return ((naive_preds[:-1, :] - true_matrix[1:, :]) ** 2).mean(axis=0)[:max_horizon]



def plot_mse_vs_naive(df, start_date = None, max_horizon=21, is_target_diff=True):
    """
    Plots model MSE vs naive baseline starting from a given date, up to max_horizon.

    Parameters:
        df (pd.DataFrame): DataFrame with 'date', 'target', and 'target_prediction' columns.
        start_date (str or pd.Timestamp): Filter rows from this date onward.
        max_horizon (int): Max forecast horizon to include (default=40).
    """
    # Filter from start_date onward
    if start_date is None:
        start_date = df.date[0]
    sub_df = df[df["date"] >= pd.to_datetime(start_date)].copy()
    if sub_df.empty:
        print(f"No data found starting from {start_date}")
        return
    
    mse_model = calculate_model_mse(sub_df, max_horizon=max_horizon)
    mse_naive = calculate_naive_mse(sub_df, is_target_diff=is_target_diff, max_horizon=max_horizon)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(mse_model) + 1), mse_model, label="Model")
    plt.plot(range(1, len(mse_naive) + 1), mse_naive, label="Naive (Last Value)")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Forecast Error vs Naive Baseline (Start: {start_date})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_aivalable_ttm_models():
    refs = list_repo_refs("ibm-granite/granite-timeseries-ttm-r2", repo_type="model")
    for branch in refs.branches:
        print(branch.ref)

def get_untrained_model(config, tsp):
    # Apply custom settings
    config.num_input_channels = tsp.num_input_channels
    config.decoder_mode = "mix_channel"
    config.prediction_channel_indices = tsp.prediction_channel_indices
    config.exogenous_channel_indices = tsp.exogenous_channel_indices
    config.enable_forecast_channel_mixing = True
    config.fcm_prepend_past = True
    untrained_model = TinyTimeMixerForPrediction(config)
    return untrained_model


def calculate_21_day_ret(df):
    df = df.copy()
    df.target_prediction = df.target_prediction.apply(lambda x:x[:21])
    df.target = df.target.apply(lambda x:x[:21])
    # Convert percentage changes to cumulative returns
    df['target_prediction'] = df['target_prediction'].apply(lambda x: np.cumprod(1 + np.array(x))[-1] - 1)
    df['target'] = df['target'].apply(lambda x: np.cumprod(1 + np.array(x))[-1] - 1)
    df = df.set_index('date')
    return df

def calculate_21_day_vol_change(df):
    """
    Calculate cumulative change in volatility over 21 days given log returns of daily volatility.
    
    Args:
        df: DataFrame with target and target_prediction columns containing arrays of log returns
        
    Returns:
        DataFrame: DataFrame with cumulative volatility changes for both target and predictions
    """
    df = df.copy()
    df.target_prediction = df.target_prediction.apply(lambda x:x[:21])
    df.target = df.target.apply(lambda x:x[:21])
    
    # Convert log returns to cumulative changes
    # For log returns, we use exp(sum) instead of cumprod(1+x)
    df['target_prediction'] = df['target_prediction'].apply(lambda x: np.exp(np.sum(np.array(x))) - 1)
    df['target'] = df['target'].apply(lambda x: np.exp(np.sum(np.array(x))) - 1)
    df = df.set_index('date')
    return df

def calculate_21_day_vol_change_mse(df):
    """
    Calculate MSE of model predictions vs naive model (variance) for 21-day volatility changes.
    To be used only for volatility log returns.
    
    Args:
        df: DataFrame with target and target_prediction columns containing arrays of log returns
        
    Returns:
        tuple: (model_mse, naive_mse)
    """
    # Make copy to avoid modifying original
    df = calculate_21_day_vol_change(df.copy())
    model_mse = ((df.target-df.target_prediction)**2).mean()
    naive_mse = df.target.var()
    
    return model_mse, naive_mse

def calculate_21_day_logret(df):
    """
    Converts arrays of daily log-returns into a single 21-day log-return,
    for both the model prediction and the true target.
    """
    df = df.copy()
    # Truncate to 21 days
    df['target_prediction'] = df['target_prediction'].apply(lambda x: np.array(x[:21]))
    df['target']            = df['target'].apply(lambda x: np.array(x[:21]))
    # Sum log-returns over the 21-day window
    df['logret_pred_21'] = df['target_prediction'].apply(lambda x: float(np.sum(x)))
    df['logret_true_21'] = df['target'].apply(lambda x: float(np.sum(x)))
    df = df.set_index('date')
    return df

def calculate_21_day_logret_mse(df):
    """
    Computes MSE of 21-day log-return forecasts versus a naive model
    that always predicts zero log-return.
    
    Returns:
        (model_mse, naive_mse)
    """
    df2 = calculate_21_day_logret(df.copy())
    # model MSE: (true_logret - pred_logret)^2
    model_mse = ((df2['logret_true_21'] - df2['logret_pred_21'])**2).mean()
    # naive MSE: variance of the true 21-day log-returns (predicting 0 every time)
    naive_mse = df2['logret_true_21'].var()
    return model_mse, naive_mse



def calculate_21_day_sum(df):
    df = df.copy()
    df.target_prediction = df.target_prediction.apply(lambda x:x[:21])
    df.target = df.target.apply(lambda x:x[:21])
    # Convert percentage changes to cumulative returns
    df['target_prediction'] = df['target_prediction'].apply(lambda x: np.sum(np.array(x)))
    df['target'] = df['target'].apply(lambda x: np.sum(np.array(x)))
    df = df.set_index('date')
    return df


def compute_performance_metrics(df_returns, benchmark_col="Close", trading_days=252):
    results = {}
    
    # Benchmark metrics
    bench = df_returns[benchmark_col]
    bench_nav = (1 + bench).cumprod()
    n_b = len(bench_nav)
    tr_b = bench_nav.iloc[-1] - 1.0
    cagr_b = bench_nav.iloc[-1] ** (trading_days / n_b) - 1.0
    vol_b = bench.std() * np.sqrt(trading_days)
    sharpe_b = bench.mean() / bench.std() * np.sqrt(trading_days)
    dd_b = bench_nav.cummax()
    mdd_b = (bench_nav / dd_b - 1.0).min()
    ir_b = 0
    
    results[benchmark_col] = {
        "Total Return":      tr_b,
        "CAGR":              cagr_b,
        "Volatility":        vol_b,
        "Sharpe":            sharpe_b,
        "Max Drawdown":      mdd_b,
        "Information Ratio": ir_b
    }
    
    # Strategy metrics
    for name, strat in df_returns.drop(columns=[benchmark_col]).items():
        sr = strat.dropna()
        nav = (1 + sr).cumprod()
        n = len(nav)
        tr = nav.iloc[-1] - 1.0
        cagr = nav.iloc[-1] ** (trading_days / n) - 1.0
        vol = sr.std() * np.sqrt(trading_days)
        sharpe = sr.mean() / sr.std() * np.sqrt(trading_days)
        dd = nav.cummax()
        mdd = (nav / dd - 1.0).min()
        aligned = pd.concat([sr, bench], axis=1, join="inner").dropna()
        diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        ir = diff.mean() / diff.std() * np.sqrt(trading_days)
        
        results[name] = {
            "Total Return":      tr,
            "CAGR":              cagr,
            "Volatility":        vol,
            "Sharpe":            sharpe,
            "Max Drawdown":      mdd,
            "Information Ratio": ir
        }
    
    perf_df = pd.DataFrame(results).T
    return perf_df