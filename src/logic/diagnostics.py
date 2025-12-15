# src/logic/diagnostics.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import io

def calculate_vif(ols_model_wrapper):
    """
    Calculates Variance Inflation Factor (VIF) for a statsmodels OLS result.
    High VIF indicates multicollinearity.
    """
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
    """
    Performs Shapiro-Wilk test for normality of residuals.
    """
    if len(residuals) > 5000:
        shapiro_stat, shapiro_p = stats.jarque_bera(residuals)
        test_name = "Jarque-Bera"
    else:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"
        
    return shapiro_stat, shapiro_p, test_name

def perform_heteroscedasticity_test(residuals, ols_model_wrapper):
    """
    Performs Breusch-Pagan test for heteroscedasticity.
    """
    exog = ols_model_wrapper.model.model.exog
    lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, exog)
    return lm_p_value

def perform_autocorrelation_test(residuals):
    """
    Performs Durbin-Watson test for autocorrelation.
    """
    dw_stat = durbin_watson(residuals)
    return dw_stat

def generate_diagnostics_report(model_wrapper):
    """
    Aggregates all diagnostic tests into a formatted text report.
    Returns: A string containing the full report.
    """
    results = model_wrapper.model
    residuals = results.resid
    
    report = []
    report.append("==================================================")
    report.append("       OLS ASSUMPTION DIAGNOSTICS REPORT          ")
    report.append("==================================================")
    report.append(f"Model: {model_wrapper.formula}")
    report.append("")
    
    # 1. Multicollinearity
    report.append("1. MULTICOLLINEARITY (Variance Inflation Factor)")
    report.append("--------------------------------------------------")
    try:
        vif_df = calculate_vif(model_wrapper)
        # Format dataframe as string table
        vif_str = vif_df.to_string(index=False, float_format="{:.4f}".format)
        report.append(vif_str)
        
        max_vif = vif_df['VIF'].max()
        if max_vif > 10:
            report.append(f"\n[WARNING] Max VIF is {max_vif:.2f} (> 10). Severe multicollinearity detected.")
            report.append("Suggestion: Center your variables or remove correlated terms.")
        elif max_vif > 5:
            report.append(f"\n[CAUTION] Max VIF is {max_vif:.2f} (> 5). Moderate multicollinearity.")
        else:
            report.append("\n[PASS] All VIF values are < 5. No severe multicollinearity.")
    except Exception as e:
        report.append(f"Could not calculate VIF: {str(e)}")
    report.append("")
    
    # 2. Normality
    report.append("2. NORMALITY OF RESIDUALS")
    report.append("--------------------------------------------------")
    stat, p_val, test_name = perform_normality_test(residuals)
    report.append(f"Test Used: {test_name}")
    report.append(f"Statistic: {stat:.4f}")
    report.append(f"p-value:   {p_val:.4f}")
    
    if p_val < 0.05:
        report.append("[FAIL] Reject H0 (p < 0.05). Residuals are NOT normally distributed.")
        report.append("Suggestion: Try log-transforming the dependent variable.")
    else:
        report.append("[PASS] Fail to reject H0 (p >= 0.05). Residuals look normal.")
    report.append("")

    # 3. Homoscedasticity
    report.append("3. HOMOSCEDASTICITY (Constant Variance)")
    report.append("--------------------------------------------------")
    lm_p = perform_heteroscedasticity_test(residuals, model_wrapper)
    report.append(f"Test Used: Breusch-Pagan")
    report.append(f"p-value:   {lm_p:.4f}")
    
    if lm_p < 0.05:
        report.append("[FAIL] Reject H0 (p < 0.05). Heteroscedasticity detected (Variance is not constant).")
        report.append("Suggestion: Use Weighted Least Squares or robust standard errors.")
    else:
        report.append("[PASS] Fail to reject H0 (p >= 0.05). Variance appears constant.")
    report.append("")

    # 4. Independence
    report.append("4. INDEPENDENCE OF ERRORS")
    report.append("--------------------------------------------------")
    dw_stat = perform_autocorrelation_test(residuals)
    report.append(f"Durbin-Watson Statistic: {dw_stat:.4f}")
    
    if 1.5 < dw_stat < 2.5:
        report.append("[PASS] Value is close to 2.0. No significant autocorrelation.")
    else:
        report.append("[WARNING] Value is far from 2.0. Possible autocorrelation.")
    report.append("")
    
    report.append("==================================================")
    report.append("End of Report")
    
    return "\n".join(report)
