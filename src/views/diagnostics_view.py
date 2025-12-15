# src/views/diagnostics_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from logic.diagnostics import (
    calculate_vif, 
    perform_normality_test, 
    perform_heteroscedasticity_test,
    perform_autocorrelation_test,
    generate_diagnostics_report
)
from logic.models import OLSWrapper

def render():
    st.subheader("🔍 OLS Assumption Diagnostics")
    st.info("This module evaluates whether your Polynomial OLS model satisfies the key statistical assumptions required for valid p-values and confidence intervals.")

    if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
        st.warning("Please load a project and run an analysis first to generate models.")
        return

    # Filter for OLS models only
    wrapped_models = st.session_state.get('wrapped_models', {})
    ols_models = {k: v for k, v in wrapped_models.items() if isinstance(v, OLSWrapper)}
    
    if not ols_models:
        st.warning("⚠️ No OLS (Polynomial Regression) models found. Diagnostics are not available for Machine Learning models like Random Forest or SVR as they do not make the same parametric assumptions.")
        return

    # Select Model
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_model_name = st.selectbox("Select OLS Model to Diagnose", list(ols_models.keys()))
    
    model_wrapper = ols_models[selected_model_name]
    
    # --- Download Report Button ---
    with col_btn:
        st.write("") # Spacing
        st.write("") 
        report_text = generate_diagnostics_report(model_wrapper)
        st.download_button(
            label="📄 Download Full Report",
            data=report_text,
            file_name=f"diagnostics_report_{selected_model_name}.txt",
            mime="text/plain",
            help="Download a text file containing the results of all diagnostic tests."
        )
    
    # Extract data from the model wrapper
    results = model_wrapper.model # statsmodels ResultsWrapper
    residuals = results.resid
    fitted_values = results.fittedvalues
    
    # --- Tabbed Interface for Diagnostics ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Multicollinearity (VIF)", 
        "2. Normality (Residuals)", 
        "3. Homoscedasticity", 
        "4. Independence"
    ])

    # --- 1. Multicollinearity (VIF) ---
    with tab1:
        st.markdown("#### Variance Inflation Factor (VIF)")
        st.markdown("""
        **Assumption:** Independent variables should not be highly correlated with each other.
        * **Goal:** VIF < 5 (Conservative) or < 10 (Relaxed).
        * **Problem:** High VIF means coefficients are unstable and p-values may be misleading.
        * **Note:** Polynomial terms ($x, x^2$) naturally cause high VIF. Consider centering data if this is critical.
        """)
        
        try:
            vif_df = calculate_vif(model_wrapper)
            
            def highlight_vif(val):
                color = 'red' if val > 10 else ('orange' if val > 5 else 'green')
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                vif_df.style.applymap(highlight_vif, subset=['VIF'])
                            .format({"VIF": "{:.2f}"}),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not calculate VIF: {e}")

    # --- 2. Normality of Residuals ---
    with tab2:
        st.markdown("#### Normality of Residuals")
        st.markdown("""
        **Assumption:** The errors (residuals) should follow a Normal distribution.
        * **Check:** Q-Q Plot points should hug the red line. Histogram should be bell-shaped.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        # Test
        stat, p_val, test_name = perform_normality_test(residuals)
        
        with col1:
            st.metric(f"{test_name} p-value", f"{p_val:.4f}")
            if p_val < 0.05:
                st.error(f"❌ **Reject H0**: Residuals are NOT normally distributed (p < 0.05). Consider transforming the response variable.")
            else:
                st.success(f"✅ **Fail to Reject H0**: Residuals appear normal (p >= 0.05).")

        # Plots
        with col2:
            # Q-Q Plot
            # Calculate quantiles
            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
            
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuals', marker=dict(color='blue', opacity=0.6)))
            
            # Regression line
            x_line = np.array([np.min(osm), np.max(osm)])
            y_line = slope * x_line + intercept
            fig_qq.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Normal Line', line=dict(color='red', width=2)))
            
            fig_qq.update_layout(
                title="Q-Q Plot", 
                xaxis_title="Theoretical Quantiles", 
                yaxis_title="Sample Quantiles",
                height=400
            )
            st.plotly_chart(fig_qq, use_container_width=True)

            # Histogram
            fig_hist = px.histogram(x=residuals, nbins=30, title="Histogram of Residuals")
            fig_hist.update_layout(xaxis_title="Residuals", showlegend=False, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- 3. Homoscedasticity ---
    with tab3:
        st.markdown("#### Homoscedasticity (Constant Variance)")
        st.markdown("""
        **Assumption:** The variance of residuals should be constant across all predicted values.
        * **Check:** Residuals vs. Fitted plot should show a random cloud, not a 'funnel' or 'U' shape.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        # Test
        lm_p = perform_heteroscedasticity_test(residuals, model_wrapper)
        
        with col1:
            st.metric("Breusch-Pagan p-value", f"{lm_p:.4f}")
            if lm_p < 0.05:
                st.error("❌ **Reject H0**: Heteroscedasticity detected (Variance is not constant).")
            else:
                st.success("✅ **Fail to Reject H0**: Homoscedasticity assumed (Variance is constant).")
        
        with col2:
            # Residuals vs Fitted Plot
            df_plot = pd.DataFrame({'Fitted': fitted_values, 'Residuals': residuals})
            fig_rvf = px.scatter(df_plot, x='Fitted', y='Residuals', title="Residuals vs. Fitted Values", opacity=0.7)
            fig_rvf.add_hline(y=0, line_dash="dash", line_color="red")
            fig_rvf.update_traces(marker=dict(size=8, color='#636EFA'))
            fig_rvf.update_layout(height=450)
            st.plotly_chart(fig_rvf, use_container_width=True)

    # --- 4. Independence ---
    with tab4:
        st.markdown("#### Independence of Errors")
        st.markdown("""
        **Assumption:** Residuals should be independent (no autocorrelation).
        """)
        
        dw_stat = perform_autocorrelation_test(residuals)
        
        st.metric("Durbin-Watson Statistic", f"{dw_stat:.4f}")
        
        if 1.5 < dw_stat < 2.5:
             st.success("✅ **No Autocorrelation**: Value is close to 2.0.")
        else:
             st.warning("⚠️ **Possible Autocorrelation**: Value is far from 2.0 (Range: 0-4).")
             
        st.info("Note: This test is most relevant for time-series data or data with a sequential order (e.g., pipetting order).")
# --- 5. Predictive Uncertainty (NEW) ---
    with tab5:
        st.markdown("### Quantifying Uncertainty")
        st.markdown("Evaluate how well the model generalizes (Cross-Validation) and how stable the coefficients are (Bootstrap).")
        
        col_cv, col_boot = st.columns(2)
        
        # --- K-Fold CV Section ---
        with col_cv:
            with st.container(border=True):
                st.markdown("#### 🔄 Cross-Validation (K-Fold)")
                st.info("Splits data into K parts to estimate 'Out-of-Sample' error.")
                k_folds = st.number_input("Number of Folds (K)", min_value=2, max_value=20, value=5)
                
                if st.button("Run K-Fold CV"):
                    with st.spinner(f"Running {k_folds}-Fold CV..."):
                        try:
                            cv_res = perform_kfold_cv(model_wrapper, k=k_folds)
                            st.write("##### Results:")
                            st.metric("Avg RMSE", f"{cv_res['avg_rmse']:.4f}", delta_color="inverse")
                            st.caption(f"(Std Dev: {cv_res['std_rmse']:.4f})")
                            st.metric("Avg R²", f"{cv_res['avg_r2']:.4f}")
                            st.caption(f"(Std Dev: {cv_res['std_r2']:.4f})")
                        except Exception as e:
                            st.error(f"CV Failed: {e}")

        # --- Bootstrap Section ---
        with col_boot:
            with st.container(border=True):
                st.markdown("#### 🎲 Bootstrap Analysis")
                st.info("Resamples data to find 95% Confidence Intervals (CI) for coefficients.")
                n_boot = st.number_input("Number of Resamples", min_value=10, max_value=1000, value=100)
                
                if st.button("Run Bootstrap"):
                    with st.spinner(f"Running {n_boot} bootstraps..."):
                        try:
                            boot_df = perform_bootstrap_analysis(model_wrapper, n_bootstraps=n_boot)
                            st.write("##### Coefficient Stability:")
                            
                            # Styling the dataframe
                            def highlight_unstable(row):
                                # If CI crosses 0 (Lower < 0 < Upper), it might be insignificant
                                if row['95% CI Lower'] < 0 and row['95% CI Upper'] > 0:
                                    return ['background-color: #fff3cd'] * len(row) # Yellow warning
                                return [''] * len(row)

                            st.dataframe(
                                boot_df[['Term', 'Original', '95% CI Lower', '95% CI Upper', 'Stable?']]
                                .style.format({
                                    'Original': '{:.4f}',
                                    '95% CI Lower': '{:.4f}',
                                    '95% CI Upper': '{:.4f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                            st.caption("⚠️ Yellow rows indicate coefficients where the 95% CI crosses zero (potentially insignificant).")
                        except Exception as e:
                            st.error(f"Bootstrap Failed: {e}")
