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

def apply_custom_layout(fig, height, width, title_size, axis_size, tick_size, 
                        show_grid=False, journal_style=False, legend_size=14,
                        title_text=None, x_text=None, y_text=None):
    """
    Applies consistent styling with High-Contrast BLACK text.
    """
    # 1. Base Layout (Force Black Fonts everywhere)
    update_dict = {
        'height': height,
        'width': width,
        'title': dict(
            font=dict(size=title_size, color='black'), # Force Black
            x=0.5,              
            xanchor='center',   
            yanchor='top'       
        ),
        'xaxis': dict(
            title_font=dict(size=axis_size, color='black'), # Force Black
            tickfont=dict(size=tick_size, color='black')    # Force Black
        ),
        'yaxis': dict(
            title_font=dict(size=axis_size, color='black'), # Force Black
            tickfont=dict(size=tick_size, color='black')    # Force Black
        ),
        'legend': dict(font=dict(size=legend_size, color='black')), # Force Black
        'margin': dict(l=80, r=40, t=80, b=80),
        'plot_bgcolor': 'white'
    }
    
    # 2. Journal Style Overrides
    if journal_style:
        axis_style = dict(
            showline=True,      
            linewidth=2,        
            linecolor='black',  
            mirror=True,        
            ticks='outside',    
            tickwidth=2,
            tickcolor='black',
            ticklen=6,
            showgrid=show_grid  
        )
        update_dict['xaxis'].update(axis_style)
        update_dict['yaxis'].update(axis_style)
    else:
        # Even in non-journal mode, keep text black but use default grid logic
        update_dict['xaxis']['showgrid'] = show_grid
        update_dict['yaxis']['showgrid'] = show_grid
    
    # 3. Apply Custom Text Labels
    if title_text is not None:
        update_dict['title']['text'] = title_text
    if x_text is not None:
        update_dict['xaxis']['title'] = dict(text=x_text, font=dict(size=axis_size, color='black'))
    if y_text is not None:
        update_dict['yaxis']['title'] = dict(text=y_text, font=dict(size=axis_size, color='black'))
        
    fig.update_layout(**update_dict)
    return fig

def render():
    st.subheader("🔍 OLS Assumption & Uncertainty Diagnostics")
    st.info("Evaluate statistical assumptions (Normality, VIF) and predictive uncertainty.")

    if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
        st.warning("Please load a project and run an analysis first to generate models.")
        return

    # --- 1. GLOBAL STYLING & EXPORT SETTINGS ---
    with st.expander("🎨 Graph Appearance & Publication Settings", expanded=False):
        st.markdown("##### 📏 Dimensions & Quality")
        c1, c2, c3 = st.columns(3)
        plot_height = c1.number_input("Height (px)", 400, 2000, 600, step=50)
        plot_width = c2.number_input("Width (px)", 400, 3000, 800, step=50, help="Set Height = Width for a square plot.")
        export_scale = c3.selectbox("Export Scale (DPI)", [1, 2, 3, 4], index=2, help="3x = 300 DPI (Print Quality)")

        st.markdown("##### ✒️ Fonts & Style")
        c4, c5, c6 = st.columns(3)
        title_font_size = c4.number_input("Title Size", 10, 50, 25)
        axis_font_size = c5.number_input("Axis Label Size", 8, 40, 25)
        tick_font_size = c6.number_input("Tick Label Size", 8, 30, 25)
        
        c7, c8, c9 = st.columns(3)
        legend_font_size = c7.number_input("Legend Text Size", 8, 30, 25)
        journal_style = c8.checkbox("Journal Style (Boxed)", value=True, help="Adds a solid black frame, removes grey grid, and points ticks outward.")
        show_grid = c9.checkbox("Show Gridlines", value=False, help="Uncheck for clean white background.")

        download_config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'high_res_plot',
                'height': plot_height,
                'width': plot_width,
                'scale': export_scale
            }
        }

    # Filter for OLS models only
    wrapped_models = st.session_state.get('wrapped_models', {})
    ols_models = {k: v for k, v in wrapped_models.items() if isinstance(v, OLSWrapper)}
    
    # Retrieve 'exp_df'
    data_df = st.session_state.get('exp_df', None) 
    independent_vars = st.session_state.get('independent_vars', [])
    
    if not ols_models:
        st.warning("⚠️ No OLS models found.")
        return

    st.divider()

    # Select Model for Interactive View
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_model_name = st.selectbox("Select Single Model to View", list(ols_models.keys()))
    
    model_wrapper = ols_models[selected_model_name]
    
    # Download Report Button
    with col_btn:
        st.write("") 
        st.write("") 
        if st.button("📄 Report (This Model)"):
             with st.spinner("Generating diagnostic report..."):
                report_text = generate_diagnostics_report(
                    model_wrapper, 
                    model_name=selected_model_name,
                    dataframe=data_df,
                    independent_vars=independent_vars
                )
                st.download_button(
                    label="Download",
                    data=report_text,
                    file_name=f"diagnostics_report_{selected_model_name}.txt",
                    mime="text/plain"
                )
    
    # Extract data from the model wrapper
    results = model_wrapper.model 
    residuals = results.resid
    fitted_values = results.fittedvalues
    y_actual = results.model.endog 
    
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
        use_centered = st.checkbox("Apply Mean-Centering (Fix Structural Multicollinearity)", value=True)

        try:
            effective_vars = independent_vars if independent_vars else model_wrapper.independent_vars

            if use_centered:
                if data_df is not None:
                    vif_df = calculate_vif(model_wrapper, dataframe=data_df, independent_vars=effective_vars)
                    st.success("✅ **Centered VIFs Active**")
                else:
                    st.warning("⚠️ Original dataset missing. Showing Raw VIFs.")
                    vif_df = calculate_vif(model_wrapper)
            else:
                st.info("ℹ️ **Raw VIFs Active**")
                vif_df = calculate_vif(model_wrapper)

            def highlight_vif(val):
                color = 'red' if val > 10 else ('orange' if val > 5 else 'green')
                return f'color: {color}; font-weight: bold'

            st.dataframe(vif_df.style.applymap(highlight_vif, subset=['VIF']).format({"VIF": "{:.2f}"}), use_container_width=True)
        except Exception as e:
            st.error(f"Could not calculate VIF: {e}")

    # --- 2. Normality of Residuals ---
    with tab2:
        st.markdown("#### Normality of Residuals")
        col1, col2 = st.columns([1, 2])
        stat, p_val, test_name = perform_normality_test(residuals)
        
        with col1:
            st.metric(f"{test_name} p-value", f"{p_val:.4f}")
            if p_val < 0.05:
                st.error("❌ **Reject H0**: Residuals NOT normal.")
            else:
                st.success("✅ **Fail to Reject H0**: Residuals look normal.")

        with col2:
            # CUSTOMIZATION FOR Q-Q PLOT
            with st.expander("✏️ Customize Q-Q Plot Labels"):
                qq_title = st.text_input("Title", "Q-Q Plot", key="qq_title")
                qq_x = st.text_input("X Label", "Theoretical Quantiles", key="qq_x")
                qq_y = st.text_input("Y Label", "Ordered Values", key="qq_y")

            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuals', marker=dict(color='black', symbol='circle-open', opacity=0.7, size=8)))
            x_line = np.array([np.min(osm), np.max(osm)])
            y_line = slope * x_line + intercept
            fig_qq.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Normal Line', line=dict(color='red', width=2)))
            
            fig_qq = apply_custom_layout(
                fig_qq, plot_height, plot_width, title_font_size, axis_font_size, tick_font_size, 
                show_grid, journal_style, legend_font_size, qq_title, qq_x, qq_y
            )
            st.plotly_chart(fig_qq, use_container_width=True, config=download_config)

            # --- Histogram Customization & Plot ---
            st.divider() # Visual separation
            with st.expander("✏️ Customize Histogram Labels"):
                hist_title = st.text_input("Title", "Residual Histogram", key="hist_title")
                hist_x = st.text_input("X Label", "Residuals", key="hist_x")
                hist_y = st.text_input("Y Label", "Count", key="hist_y")

            fig_hist = px.histogram(x=residuals, nbins=30, title=hist_title, color_discrete_sequence=['lightgrey'])
            fig_hist.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=0.8)
            
            fig_hist = apply_custom_layout(
                fig_hist, plot_height, plot_width, title_font_size, axis_font_size, tick_font_size, 
                show_grid, journal_style, legend_font_size, hist_title, hist_x, hist_y
            )
            st.plotly_chart(fig_hist, use_container_width=True, config=download_config)

    # --- 3. Homoscedasticity ---
    with tab3:
        st.markdown("#### Homoscedasticity")
        col1, col2 = st.columns([1, 2])
        lm_p = perform_heteroscedasticity_test(residuals, model_wrapper)
        
        with col1:
            st.metric("Breusch-Pagan p-value", f"{lm_p:.4f}")
            if lm_p < 0.05:
                st.error("❌ Heteroscedasticity detected.")
            else:
                st.success("✅ Variance is constant.")
        
        with col2:
            with st.expander("✏️ Customize Plot Labels"):
                rvf_title = st.text_input("Title", "Residuals vs. Fitted Values", key="rvf_title")
                rvf_x = st.text_input("X Label", "Fitted Values", key="rvf_x")
                rvf_y = st.text_input("Y Label", "Residuals", key="rvf_y")

            df_plot = pd.DataFrame({'Fitted': fitted_values, 'Residuals': residuals})
            fig_rvf = px.scatter(df_plot, x='Fitted', y='Residuals', opacity=0.7)
            fig_rvf.add_hline(y=0, line_dash="dash", line_color="red")
            fig_rvf.update_traces(marker=dict(size=8, color='black', symbol='circle-open')) 
            
            fig_rvf = apply_custom_layout(
                fig_rvf, plot_height, plot_width, title_font_size, axis_font_size, tick_font_size, 
                show_grid, journal_style, legend_font_size, rvf_title, rvf_x, rvf_y
            )
            st.plotly_chart(fig_rvf, use_container_width=True, config=download_config)

    # --- 4. Independence ---
    with tab4:
        st.markdown("#### Independence of Errors")
        dw_stat = perform_autocorrelation_test(residuals)
        st.metric("Durbin-Watson Statistic", f"{dw_stat:.4f}")
        
        with st.expander("✏️ Customize Plot Labels"):
            ind_title = st.text_input("Title", "Residuals vs. Experiment Order", key="ind_title")
            ind_x = st.text_input("X Label", "Experiment Order (Index)", key="ind_x")
            ind_y = st.text_input("Y Label", "Residuals", key="ind_y")

        fig_order = px.scatter(y=residuals)
        fig_order.add_hline(y=0, line_dash="dash", line_color="red")
        fig_order.update_traces(mode='lines+markers', marker=dict(color='black', size=6), line=dict(color='grey', width=1)) 
        
        fig_order = apply_custom_layout(
            fig_order, plot_height, plot_width, title_font_size, axis_font_size, tick_font_size, 
            show_grid, journal_style, legend_font_size, ind_title, ind_x, ind_y
        )
        st.plotly_chart(fig_order, use_container_width=True, config=download_config)

        if 1.5 < dw_stat < 2.5:
             st.success("✅ No Autocorrelation")
        else:
             st.warning("⚠️ Possible Autocorrelation")

    # --- 5. Predictive Uncertainty ---
    with tab5:
        st.markdown("### Quantifying Uncertainty")
        st.markdown("#### Predicted vs. Observed")
        
        with st.expander("✏️ Customize Plot Labels", expanded=True):
            pred_title = st.text_input("Title", "Predicted vs. Observed", key="pred_title")
            pred_x = st.text_input("X Label", "Observed (Actual)", key="pred_x")
            pred_y = st.text_input("Y Label", "Predicted (Model)", key="pred_y")

        col_plot, _ = st.columns([2, 1])
        with col_plot:
            min_val = min(min(y_actual), min(fitted_values))
            max_val = max(max(y_actual), max(fitted_values))
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_actual, y=fitted_values, mode='markers', 
                name='Data', marker=dict(color='black', opacity=0.6, size=8, symbol='circle-open')
            ))
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], 
                mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')
            ))
            
            fig_pred = apply_custom_layout(
                fig_pred, plot_height, plot_width, title_font_size, axis_font_size, tick_font_size, 
                show_grid, journal_style, legend_font_size, pred_title, pred_x, pred_y
            )
            st.plotly_chart(fig_pred, use_container_width=True, config=download_config)

        st.divider()

        col_cv, col_boot = st.columns(2)
        with col_cv:
            with st.container(border=True):
                st.markdown("#### 🔄 Cross-Validation (K-Fold)")
                k_folds = st.number_input("Number of Folds (K)", min_value=2, max_value=20, value=5)
                if st.button("Run K-Fold CV"):
                    try:
                        cv_res = perform_kfold_cv(model_wrapper, k=k_folds)
                        st.write("##### Results:")
                        st.metric("Avg RMSE", f"{cv_res['avg_rmse']:.4f}")
                        st.metric("Avg R²", f"{cv_res['avg_r2']:.4f}")
                    except Exception as e:
                        st.error(f"CV Failed: {e}")

        with col_boot:
            with st.container(border=True):
                st.markdown("#### 🎲 Bootstrap Analysis")
                n_boot = st.number_input("Number of Resamples", min_value=10, max_value=1000, value=1000)
                if st.button("Run Bootstrap"):
                    try:
                        boot_df = perform_bootstrap_analysis(model_wrapper, n_bootstraps=n_boot)
                        st.dataframe(boot_df[['Term', 'Original', '95% CI Lower', '95% CI Upper', 'Stable?']].style.format({'Original': '{:.4f}','95% CI Lower': '{:.4f}','95% CI Upper': '{:.4f}'}), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Bootstrap Failed: {e}")
