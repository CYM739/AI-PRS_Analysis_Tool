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
    perform_kfold_cv,
    perform_bootstrap_analysis,
    generate_diagnostics_report,
    generate_full_project_report
)
from logic.models import OLSWrapper

def render():
    st.subheader("🔍 OLS Assumption & Uncertainty Diagnostics")
    st.info("This module evaluates statistical assumptions and quantifies predictive uncertainty using Cross-Validation and Bootstrapping.")

    if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
        st.warning("Please load a project and run an analysis first to generate models.")
        return

    # Filter for OLS models only
    wrapped_models = st.session_state.get('wrapped_models', {})
    ols_models = {k: v for k, v in wrapped_models.items() if isinstance(v, OLSWrapper)}
    data_df = st.session_state.get('data_df', None)
    
    # Robustly get independent vars from session state (Source of Truth)
    independent_vars = st.session_state.get('independent_vars', [])
    
    if not ols_models:
        st.warning("⚠️ No OLS (Polynomial Regression) models found. Diagnostics are not available for Machine Learning models like Random Forest or SVR as they do not make the same parametric assumptions.")
        return

    # --- TOP LEVEL ACTIONS ---
    col_dl_all, _ = st.columns([1, 2])
    with col_dl_all:
        if st.button("📥 Download Diagnostics for ALL Models"):
            with st.spinner("Generating combined report for all OLS models..."):
                full_project_report = generate_full_project_report(wrapped_models)
                st.download_button(
                    label="📄 Save Full Project Report",
                    data=full_project_report,
                    file_name="full_project_diagnostics_report.txt",
                    mime="text/plain",
                    help="Download a single text file containing diagnostic results for every OLS model in the project."
                )
    st.divider()

    # Select Model for Interactive View
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_model_name = st.selectbox("Select Single Model to View", list(ols_models.keys()))
    
    model_wrapper = ols_models[selected_model_name]
    
    # --- Download Single Report Button ---
    with col_btn:
        st.write("") # Spacing
        st.write("") 
        if st.button("📄 Report (This Model)"):
             with st.spinner("Generating diagnostic report..."):
                report_text = generate_diagnostics_report(model_wrapper, model_name=selected_model_name)
                st.download_button(
                    label="Download",
                    data=report_text,
                    file_name=f"diagnostics_report_{selected_model_name}.txt",
                    mime="text/plain"
                )
    
    # Extract data from the model wrapper
    results = model_wrapper.model # statsmodels ResultsWrapper
    residuals = results.resid
    fitted_values = results.fittedvalues
    
    # --- Tabbed Interface for Diagnostics ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Multicollinearity", 
        "2. Normality", 
        "3. Homoscedasticity", 
        "4. Independence",
        "5. Predictive Uncertainty"
    ])

# --- 1. Multicollinearity (VIF) ---
    with tab1:
        st.markdown("#### Variance Inflation Factor (VIF)")
        
        # Checkbox
        use_centered = st.checkbox(
            "Apply Mean-Centering (Fix Structural Multicollinearity)", 
            value=True, 
            help="Subtracts the mean from doses before calculating VIF. This removes artificial correlation between Dose and Dose^2."
        )

        try:
            # Prepare variable list
            effective_vars = independent_vars if independent_vars else model_wrapper.independent_vars

            # LOGIC SPLIT: Explicitly handle missing data case
            if use_centered:
                if data_df is not None:
                    # ---------------------------------------------------------
                    # SHOW PREVIEW (Only if data exists)
                    # ---------------------------------------------------------
                    df_display = data_df.copy()
                    means_dict = {}
                    
                    # Perform centering for display
                    for var in effective_vars:
                        if var in df_display.columns and pd.api.types.is_numeric_dtype(df_display[var]):
                            mean_val = df_display[var].mean()
                            means_dict[var] = mean_val
                            df_display[var] = df_display[var] - mean_val

                    # Force expanded=True so you definitely see it
                    with st.expander("🔎 Inspect Centered Data (Debugging)", expanded=True):
                        st.caption("These are the values being used for VIF calculation. They should be centered around 0.")
                        st.write("**Means subtracted:**", means_dict)
                        # Show only relevant columns to avoid clutter
                        cols_to_show = [c for c in effective_vars if c in df_display.columns]
                        st.dataframe(df_display[cols_to_show].head(), use_container_width=True)

                    # Calculate Centered VIF
                    vif_df = calculate_vif(
                        model_wrapper, 
                        dataframe=data_df, 
                        independent_vars=effective_vars
                    )
                    st.success("✅ **Centered VIFs Active**")
                    
                else:
                    # Data is missing, but user asked for centering
                    st.warning("⚠️ **Cannot Apply Centering**: The original dataset (`data_df`) is missing from the session state. Showing Raw VIFs instead.")
                    vif_df = calculate_vif(model_wrapper)

            else:
                # User unchecked the box
                st.info("ℹ️ **Raw VIFs Active**: High values are expected for polynomial terms.")
                vif_df = calculate_vif(model_wrapper)

            # Display VIF Table
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
        st.caption("Checks if errors follow a Bell Curve (p > 0.05 is good).")
        
        col1, col2 = st.columns([1, 2])
        
        # Test
        stat, p_val, test_name = perform_normality_test(residuals)
        
        with col1:
            st.metric(f"{test_name} p-value", f"{p_val:.4f}")
            if p_val < 0.05:
                st.error("❌ **Reject H0**: Residuals NOT normal.")
            else:
                st.success("✅ **Fail to Reject H0**: Residuals look normal.")

        # Plots
        with col2:
            # Q-Q Plot
            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuals', marker=dict(color='blue', opacity=0.6)))
            x_line = np.array([np.min(osm), np.max(osm)])
            y_line = slope * x_line + intercept
            fig_qq.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Normal Line', line=dict(color='red', width=2)))
            fig_qq.update_layout(title="Q-Q Plot", height=350)
            st.plotly_chart(fig_qq, use_container_width=True)

            # Histogram
            fig_hist = px.histogram(x=residuals, nbins=30, title="Residual Histogram")
            fig_hist.update_layout(xaxis_title="Residuals", showlegend=False, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- 3. Homoscedasticity ---
    with tab3:
        st.markdown("#### Homoscedasticity (Constant Variance)")
        st.caption("Checks if error variance is constant (p > 0.05 is good).")
        
        col1, col2 = st.columns([1, 2])
        
        # Test
        lm_p = perform_heteroscedasticity_test(residuals, model_wrapper)
        
        with col1:
            st.metric("Breusch-Pagan p-value", f"{lm_p:.4f}")
            if lm_p < 0.05:
                st.error("❌ **Reject H0**: Heteroscedasticity detected.")
            else:
                st.success("✅ **Fail to Reject H0**: Variance is constant.")
        
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
        st.caption("Checks if errors are correlated with each other (Target ~2.0).")
        
        dw_stat = perform_autocorrelation_test(residuals)
        
        st.metric("Durbin-Watson Statistic", f"{dw_stat:.4f}")
        
        if 1.5 < dw_stat < 2.5:
             st.success("✅ **No Autocorrelation**: Value is close to 2.0.")
        else:
             st.warning("⚠️ **Possible Autocorrelation**: Value is far from 2.0 (Range: 0-4).")
             
        st.info("Note: This test is most relevant for time-series data or data with a sequential order.")

    # --- 5. Predictive Uncertainty ---
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
                n_boot = st.number_input("Number of Resamples", min_value=10, max_value=1000, value=1000)
                
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
