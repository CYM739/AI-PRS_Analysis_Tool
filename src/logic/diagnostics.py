# src/logic/diagnostics.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

def calculate_vif(ols_model_wrapper):
    """
    Calculates Variance Inflation Factor (VIF) for a statsmodels OLS result.
    High VIF indicates multicollinearity.
    """
    # Access the design matrix (exog) from the statsmodels object
    # wrapper.model is RegressionResultsWrapper -> .model is OLS Model -> .exog is the design matrix
    exog = ols_model_wrapper.model.model.exog
    exog_names = ols_model_wrapper.model.model.exog_names
    
    vif_data = []
    for i, name in enumerate(exog_names):
        # We generally skip the Intercept for VIF interpretation
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
    H0: Data is normally distributed.
    p < 0.05 rejects H0 (Not normal).
    """
    # Shapiro-Wilk is suitable for N < 5000. For larger N, it might be too sensitive,
    # but strictly speaking, it's the standard test.
    if len(residuals) > 5000:
        # Fallback to Jarque-Bera for large datasets if needed, but keeping simple for now
        shapiro_stat, shapiro_p = stats.jarque_bera(residuals)
        test_name = "Jarque-Bera"
    else:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"
        
    return shapiro_stat, shapiro_p, test_name

def perform_heteroscedasticity_test(residuals, ols_model_wrapper):
    """
    Performs Breusch-Pagan test for heteroscedasticity.
    H0: Homoscedasticity (Variance is constant).
    p < 0.05 rejects H0 (Variance is not constant).
    """
    exog = ols_model_wrapper.model.model.exog
    lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, exog)
    return lm_p_value

def perform_autocorrelation_test(residuals):
    """
    Performs Durbin-Watson test for autocorrelation.
    Range: 0 - 4. 
    ~2.0 is No Autocorrelation.
    < 1.5 is Positive Autocorrelation.
    > 2.5 is Negative Autocorrelation.
    """
    dw_stat = durbin_watson(residuals)
    return dw_stat
