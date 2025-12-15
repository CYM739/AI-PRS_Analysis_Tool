# src/logic/diagnostics.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample

def calculate_vif(ols_model_wrapper):
    """Calculates Variance Inflation Factor (VIF)."""
    exog = ols_model_wrapper.model.model.exog
    exog_names = ols_model_wrapper.model.model.exog_names
    
    vif_data = []
    for i, name in enumerate(exog_names):
        if name.lower() == 'intercept':
            continue
        try:
            vif = variance_inflation_factor(exog, i)
            vif_data.append({'Feature': name, 'VIF': vif})
        except Exception:
            vif_data.append({'Feature': name, 'VIF': np.nan})
        
    return pd.DataFrame(vif_data)

def perform_normality_test(residuals):
    """Performs Shapiro-Wilk or Jarque-Bera test."""
    if len(residuals) > 5000:
        stat, p_val = stats.jarque_bera(residuals)
        test_name = "Jarque-Bera"
    else:
        stat, p_val = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"
    return stat, p_val, test_name

def perform_heteroscedasticity_test(residuals, ols_model_wrapper):
    """Performs Breusch-Pagan test."""
    exog = ols_model_wrapper.model.model.exog
    lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, exog)
    return lm_p_value

def perform_autocorrelation_test(residuals):
    """Performs Durbin-Watson test."""
    return durbin_watson(residuals)

# --- NEW FUNCTIONS FOR UNCERTAINTY QUANTIFICATION ---

def perform_kfold_cv(model_wrapper, k=5):
    """
    Performs K-Fold Cross-Validation to estimate out-of-sample performance.
    """
    results = model_wrapper.model
    # statsmodels stores the design matrix in results.model.exog/endog
    X = results.model.exog
    y = results.model.endog
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    cv_scores = {'mse': [], 'rmse': [], 'r2': []}
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Refit OLS on training fold
        # Note: We fit directly using statsmodels OLS on the design matrix
        model_fold = sm.OLS(y_train, X_train).fit()
        preds = model_fold.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        cv_scores['mse'].append(mse)
        cv_scores['rmse'].append(np.sqrt(mse))
        cv_scores['r2'].append(r2_score(y_test, preds))
        
    return {
        'k': k,
        'avg_mse': np.mean(cv_scores['mse']),
        'std_mse': np.std(cv_scores['mse']),
        'avg_rmse': np.mean(cv_scores['rmse']),
        'std_rmse': np.std(cv_scores['rmse']),
        'avg_r2': np.mean(cv_scores['r2']),
        'std_r2': np.std(cv_scores['r2'])
    }

def perform_bootstrap_analysis(model_wrapper, n_bootstraps=100):
    """
    Performs Bootstrap resampling to estimate coefficient stability and confidence intervals.
    """
    results = model_wrapper.model
    X = results.model.exog
    y = results.model.endog
    exog_names = results.model.exog_names
    original_params = results.params
    
    boot_params = []
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        X_res, y_res = resample(X, y, random_state=None)
        try:
            model_boot = sm.OLS(y_res, X_res).fit()
            boot_params.append(model_boot.params)
        except:
            continue # Skip failed fits
            
    boot_df = pd.DataFrame(boot_params)
    
    stats_list = []
    # Columns in boot_df usually correspond to indices of exog_names
    for i, name in enumerate(exog_names):
        if i >= boot_df.shape[1]: continue
        
        values = boot_df.iloc[:, i]
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        orig = original_params[i] if i < len(original_params) else 0
        
        stats_list.append({
            'Term': name,
            'Original': orig,
            'Boot Mean': values.mean(),
            '95% CI Lower': lower,
            '95% CI Upper': upper,
            'Stable?': "✅" if (lower < 0 < upper) is False else "⚠️" # Check if 0 is in CI
        })
        
    return pd.DataFrame(stats_list)

def generate_diagnostics_report(model_wrapper):
    """
    Aggregates all diagnostic tests into a formatted text report.
    Includes Quick CV and Bootstrap results.
    """
    results = model_wrapper.model
    residuals = results.resid
    
    report = []
    report.append("==================================================")
    report.append("       OLS ASSUMPTION & UNCERTAINTY REPORT        ")
    report.append("==================================================")
    report.append(f"Model: {model_wrapper.formula}")
    report.append("")
    
    # 1. Multicollinearity
    report.append("1. MULTICOLLINEARITY (VIF)")
    report.append("--------------------------------------------------")
    try:
        vif_df = calculate_vif(model_wrapper)
        report.append(vif_df.to_string(index=False, float_format="{:.4f}".format))
    except Exception as e:
        report.append(f"Error: {e}")
    report.append("")
    
    # 2. Normality
    report.append("2. NORMALITY OF RESIDUALS")
    report.append("--------------------------------------------------")
    stat, p_val, test_name = perform_normality_test(residuals)
    report.append(f"Test: {test_name}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        report.append("[FAIL] Residuals are NOT normal.")
    else:
        report.append("[PASS] Residuals appear normal.")
    report.append("")

    # 3. Homoscedasticity
    report.append("3. HOMOSCEDASTICITY")
    report.append("--------------------------------------------------")
    lm_p = perform_heteroscedasticity_test(residuals, model_wrapper)
    report.append(f"Breusch-Pagan p-value: {lm_p:.4f}")
    if lm_p < 0.05:
        report.append("[FAIL] Heteroscedasticity detected.")
    else:
        report.append("[PASS] Variance appears constant.")
    report.append("")

    # 4. Independence
    report.append("4. INDEPENDENCE (Autocorrelation)")
    report.append("--------------------------------------------------")
    dw_stat = perform_autocorrelation_test(residuals)
    report.append(f"Durbin-Watson: {dw_stat:.4f}")
    report.append("")

    # 5. Predictive Uncertainty (New)
    report.append("5. PREDICTIVE UNCERTAINTY (CV & Bootstrap)")
    report.append("--------------------------------------------------")
    
    # Quick CV (K=5)
    report.append("(A) 5-Fold Cross-Validation Results:")
    try:
        cv_res = perform_kfold_cv(model_wrapper, k=5)
        report.append(f"    Avg RMSE: {cv_res['avg_rmse']:.4f} (+/- {cv_res['std_rmse']:.4f})")
        report.append(f"    Avg R2:   {cv_res['avg_r2']:.4f} (+/- {cv_res['std_r2']:.4f})")
    except Exception as e:
        report.append(f"    CV Error: {e}")
    report.append("")

    # Quick Bootstrap (N=50)
    report.append("(B) Bootstrap Analysis (N=50) - Coefficient Stability:")
    try:
        boot_df = perform_bootstrap_analysis(model_wrapper, n_bootstraps=50)
        # Select columns to display
        disp_cols = ['Term', 'Original', '95% CI Lower', '95% CI Upper', 'Stable?']
        report.append(boot_df[disp_cols].to_string(index=False, float_format="{:.4f}".format))
        report.append("\n    Note: 'Stable?' checks if 95% CI excludes 0 (Significant).")
        report.append("          ⚠️ means CI crosses 0 (likely insignificant).")
    except Exception as e:
        report.append(f"    Bootstrap Error: {e}")

    report.append("")
    report.append("==================================================")
    report.append("End of Report")
    
    return "\n".join(report)
