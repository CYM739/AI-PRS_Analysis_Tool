# src/views/evaluation_view.py
import streamlit as st
import pandas as pd
from logic.plotting import plot_actual_vs_predicted
from logic.models import OLSWrapper
from utils.ui_helpers import format_variable_options

def render():
    """Renders all UI components and logic for the Model Evaluation tab."""
    st.subheader("✅ Compare Actual vs. Predicted Data")

    # MODIFICATION: Check if models exist before rendering the selectbox
    if "wrapped_models" not in st.session_state or not st.session_state.wrapped_models:
        st.info("Run an analysis to see model evaluations.")
        return

    # Use wrapped_models instead of ols_results
    formatted_models_eval = format_variable_options(st.session_state.wrapped_models.keys())
    selected_model_eval_formatted = st.selectbox("Select Model to Evaluate", options=formatted_models_eval, key="eval_model")

    if not selected_model_eval_formatted:
        # This condition might now be redundant, but it's safe to keep.
        st.info("Run an analysis to see model evaluations.")
        return

    model_to_eval = selected_model_eval_formatted.split(":")[0]

    with st.expander("🎨 Customize Plot Appearance"):
        st.write("**Plot Aesthetics:**")
        c1, c2, c3 = st.columns(3)
        font_size = c1.slider("Font Size", 6, 20, 10)
        bar_width = c2.slider("Bar Width", 0.1, 0.8, 0.35)
        dot_size = c3.slider("Dot Size", 5, 200, 30)

        st.write("---")
        st.write("**Plot Titles (Scatter Plot):**")
        custom_main_title_tab3 = st.text_input("Main Title", placeholder="Leave blank for default", key="tab3_main_title")
        custom_x_title_tab3 = st.text_input("X-axis Title", placeholder="Leave blank for default", key="tab3_x_title")
        custom_y_title_tab3 = st.text_input("Y-axis Title", placeholder="Leave blank for default", key="tab3_y_title")

    try:
        # Get the wrapped model object directly
        selected_model = st.session_state.wrapped_models[model_to_eval]

        # This logic needs to adapt based on model type
        # For OLSWrapper, we can still access the underlying model details
        if isinstance(selected_model, OLSWrapper):
             y_predicted = selected_model.model.fittedvalues
             y_actual = pd.Series(selected_model.model.model.endog, index=y_predicted.index, name=model_to_eval)
        else:
            # For other models (like SVR and RandomForest), we predict on the clean training data
            df_for_prediction = st.session_state.expanded_df if isinstance(selected_model, OLSWrapper) else st.session_state.exp_df
            clean_df = df_for_prediction.dropna(subset=[model_to_eval] + st.session_state.independent_vars).copy()
            y_actual = clean_df[model_to_eval]
            y_predicted = selected_model.predict(clean_df)


    except Exception as e:
        st.error(f"An error occurred while retrieving model data: {e}")
        return

    if y_actual.empty or len(y_predicted) == 0:
        st.warning("No valid data points available in the fitted model to plot.")
    else:
        # Ensure y_predicted is a pandas Series for consistency
        if not isinstance(y_predicted, pd.Series):
            y_predicted = pd.Series(y_predicted, index=y_actual.index)

        fig1, fig2 = plot_actual_vs_predicted(
            y_actual, y_predicted, model_to_eval,
            variable_descriptions=st.session_state.variable_descriptions,
            main_title=custom_main_title_tab3 or None, x_title=custom_x_title_tab3 or None,
            y_title=custom_y_title_tab3 or None, font_size=font_size, bar_width=bar_width, dot_size=dot_size
        )
        st.write("#### Scatter Plot Comparison")
        st.pyplot(fig2)
        st.write("#### Bar Chart Comparison")
        st.pyplot(fig1)