# src/views/synergy_view.py
import streamlit as st
import pandas as pd
import numpy as np
from logic.data_processing import calculate_synergy
from logic.plotting import plot_synergy_heatmap
from utils.ui_helpers import format_variable_options
from utils.state_management import clear_synergy_results

def validate_synergy_data(df, drug1_col, drug2_col):
    """
    Checks if the dataframe contains the necessary single-agent control data.
    """
    # Convert to numeric for validation to match processing logic
    d1_vals = pd.to_numeric(df[drug1_col], errors='coerce').fillna(0)
    d2_vals = pd.to_numeric(df[drug2_col], errors='coerce').fillna(0)
    
    # Check for 0 presence
    has_drug1_zero = (np.abs(d1_vals) < 1e-6).any()
    has_drug2_zero = (np.abs(d2_vals) < 1e-6).any()
    
    if not has_drug1_zero or not has_drug2_zero:
        st.error(f"⚠️ **Critical Error:** Missing '0' dose control. The dataset must contain data where {drug1_col}=0 and {drug2_col}=0.")
        return False
        
    # Check for Single Agent Rows (Rows where one drug is >0 and other is 0)
    # This detects the "Only combination points" problem
    d1_single_agent = df[(d1_vals > 1e-6) & (np.abs(d2_vals) < 1e-6)]
    d2_single_agent = df[(np.abs(d1_vals) < 1e-6) & (d2_vals > 1e-6)]

    if d1_single_agent.empty or d2_single_agent.empty:
        missing = []
        if d1_single_agent.empty: missing.append(f"Single-agent {drug1_col} (where {drug2_col}=0)")
        if d2_single_agent.empty: missing.append(f"Single-agent {drug2_col} (where {drug1_col}=0)")
        
        st.warning(f"""
        ⚠️ **Data Warning: Missing Single-Agent Arms**
        Your data seems to contain combinations, but is missing the pure single-agent experiments:
        - Missing: **{', '.join(missing)}**
        
        Without these reference points, the synergy calculation (which compares Combination vs Single Agent) will result in empty (NaN) cells for most of the matrix.
        """)
        # We don't return False here because maybe they only have partial data, but we warn strongly.
        
    return True

def render():
    """Renders all UI components and logic for the Synergy Analysis tab."""
    st.subheader("🤝 Drug Combination Synergy Analysis")

    if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
        st.info("Load a project and run or load an analysis from the 'Project Library' to use this tool.")
        return
        
    st.info("""
    This tool calculates synergy based on validated pharmacological models (Nature Communications 2025).
    Select the model below to see interpretation guidelines.
    """)
    
    with st.container(border=True):
        st.markdown("##### Configuration")
        col1, col2, col3 = st.columns(3)

        drug_options = format_variable_options(st.session_state.independent_vars)
        drug1_formatted = col1.selectbox("Select Drug 1 (Y-axis)", options=drug_options, key="synergy_drug1", on_change=clear_synergy_results)
        drug1 = drug1_formatted.split(":")[0]
        
        drug2_options = [opt for opt in drug_options if not opt.startswith(drug1)]
        drug2_formatted = col2.selectbox("Select Drug 2 (X-axis)", options=drug2_options, key="synergy_drug2", on_change=clear_synergy_results)
        drug2 = drug2_formatted.split(":")[0]

        effect_options = format_variable_options(st.session_state.dependent_vars)
        effect_formatted = col3.selectbox("Select Effect Variable", options=effect_options, key="synergy_effect", on_change=clear_synergy_results)
        effect = effect_formatted.split(":")[0]

        transform_data = st.checkbox("Data is cell viability (0-1); transform to inhibition (1-viability).", value=True, key="transform_toggle", on_change=clear_synergy_results)
        if transform_data:
            st.markdown(f"The analysis will use `1 - {effect}` as the measure of inhibition.")
        else:
             st.markdown(f"The analysis will use the raw '{effect}' values. Ensure higher values represent greater drug effect (Inhibition).")

        synergy_model_display = st.selectbox(
            "Select Synergy Model",
            ["Gamma (Recommended)", "HSA (Excess Highest Single Agent)"],
            on_change=clear_synergy_results
        )
        synergy_model_key = synergy_model_display.split(" ")[0].lower()

        if synergy_model_key == 'gamma':
            st.success("""
            **Gamma Score Interpretation (Ratio):**
            * **< 0.95:** Synergistic.
            * **~ 1.0:** Additive.
            * **> 1.0:** Antagonistic.
            """)
        elif synergy_model_key == 'hsa':
            st.success("""
            **Excess HSA Interpretation (Difference):**
            * **> 0:** Synergistic.
            * **< 0:** Antagonistic.
            """)

        validate_synergy_data(st.session_state.exp_df, drug1, drug2)

    if st.button("Calculate Synergy", type="primary"):
        with st.spinner("Calculating synergy scores..."):
            try:
                df_synergy = st.session_state.exp_df.copy()
                
                if transform_data:
                    if df_synergy[effect].max() > 1.0:
                         st.warning("Data appears to be in percentage (0-100). Treating as 0-1 fraction.")
                    df_synergy[effect] = 1 - df_synergy[effect]

                synergy_matrix = calculate_synergy(
                    dataframe=df_synergy,
                    drug1_name=drug1,
                    drug2_name=drug2,
                    effect_name=effect,
                    model=synergy_model_key
                )
                
                st.session_state.synergy_matrix = synergy_matrix
                st.session_state.synergy_drugs = (drug1, drug2)
                st.session_state.synergy_model_name = synergy_model_display

            except Exception as e:
                st.error(f"An error occurred during calculation: {e}")
                clear_synergy_results()
    
    if st.session_state.get('synergy_matrix') is not None and not st.session_state.synergy_matrix.empty:
        st.write("---")
        synergy_matrix = st.session_state.synergy_matrix
        drug1, drug2 = st.session_state.synergy_drugs
        model_name = st.session_state.get('synergy_model_name', 'Synergy')
        
        # Check for empty calculations
        nan_count = synergy_matrix.isna().sum().sum()
        total_cells = synergy_matrix.size
        if nan_count > 0:
            st.warning(f"⚠️ **Incomplete Result:** {nan_count}/{total_cells} cells are empty (NaN). This confirms that Single-Agent reference data for those specific dose combinations was missing from the input.")

        with st.expander("🔢 Raw Synergy Data (Matrix)", expanded=True):
            st.dataframe(synergy_matrix, use_container_width=True)
            csv = synergy_matrix.to_csv().encode('utf-8')
            st.download_button(
                label="Download Synergy Matrix as CSV",
                data=csv,
                file_name=f'synergy_matrix_{model_name}.csv',
                mime='text/csv',
            )

        st.subheader("Synergy Heatmap")
        descriptions = st.session_state.variable_descriptions
        drug1_desc = descriptions.get(drug1, drug1)
        drug2_desc = descriptions.get(drug2, drug2)

        fig = plot_synergy_heatmap(synergy_matrix, drug1_desc, drug2_desc, model_name)
        st.plotly_chart(fig, use_container_width=True)
