# src/views/synergy_view.py
import streamlit as st
import pandas as pd
from logic.data_processing import calculate_synergy
from logic.plotting import plot_synergy_heatmap
from utils.ui_helpers import format_variable_options
from utils.state_management import clear_synergy_results

def validate_synergy_data(df, drug1_col, drug2_col):
    """
    Checks if the dataframe contains the necessary single-agent control data.
    """
    drug1_doses = df[drug1_col].unique()
    drug2_doses = df[drug2_col].unique()
    
    # Check if a '0' dose exists for both drugs
    has_drug1_control = 0 in drug1_doses
    has_drug2_control = 0 in drug2_doses
    
    if not has_drug1_control or not has_drug2_control:
        missing = []
        if not has_drug1_control:
            missing.append(f"a zero-dose row for '{drug1_col}'")
        if not has_drug2_control:
            missing.append(f"a zero-dose row for '{drug2_col}'")
        
        st.warning(f"⚠️ **Data Validation Warning:** The synergy calculation requires single-agent controls. Your dataset appears to be missing { ' and '.join(missing) }. The plot may be empty or incorrect.")
        return False
    return True


def render():
    """Renders all UI components and logic for the Synergy Analysis tab."""
    st.subheader("🤝 Drug Combination Synergy Analysis")

    if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
        st.info("Load a project and run or load an analysis from the 'Project Library' to use this tool.")
        return
        
    st.info("""
    This tool calculates synergy based on established pharmacological models.
    - **Synergy (Red areas):** The combination is more effective than expected.
    - **Antagonism (Blue areas):** The combination is less effective than expected.
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
             st.markdown(f"The analysis will use the raw '{effect}' values. Ensure higher values represent greater drug effect.")

        synergy_model_display = st.selectbox(
            "Select Synergy Model",
            ["Gamma (via Excess HSA)", "HSA (Highest Single Agent)", "Loewe Additivity (placeholder)"],
            on_change=clear_synergy_results
        )
        synergy_model_key = synergy_model_display.split(" ")[0].lower()

        # Perform data validation before showing the button
        validate_synergy_data(st.session_state.exp_df, drug1, drug2)

    if st.button("Calculate Synergy", type="primary"):
        with st.spinner("Calculating synergy scores..."):
            try:
                df_synergy = st.session_state.exp_df.copy()
                if transform_data:
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
                st.error("Please ensure your data is formatted correctly for a synergy matrix.")
                clear_synergy_results()
    
    if st.session_state.get('synergy_matrix') is not None and not st.session_state.synergy_matrix.empty:
        st.write("---")
        st.subheader("Synergy Heatmap")
        synergy_matrix = st.session_state.synergy_matrix
        drug1, drug2 = st.session_state.synergy_drugs
        model_name = st.session_state.get('synergy_model_name', 'Synergy')

        descriptions = st.session_state.variable_descriptions
        drug1_desc = descriptions.get(drug1, drug1)
        drug2_desc = descriptions.get(drug2, drug2)

        fig = plot_synergy_heatmap(synergy_matrix, drug1_desc, drug2_desc, model_name)
        st.plotly_chart(fig, use_container_width=True)