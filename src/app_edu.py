# src/app_edu.py
import streamlit as st
import pandas as pd
from utils import state_management
from views_edu import dashboard_view
from views.library_view import load_library
from logic.data_processing import analyze_csv, expand_terms, run_analysis
from io import StringIO

def auto_load_and_analyze_project():
    """Automatically loads the first project and runs analysis if no project is active."""
    if st.session_state.get('processed_file'):
        return

    library = load_library()
    if not library:
        return

    # Load the first project found
    project_name = sorted(list(library.keys()))[0]
    project_data = library[project_name]
    df = pd.read_json(StringIO(project_data['data_df_json']), orient='split')

    # Set project data in session state
    (st.session_state.exp_df, st.session_state.all_vars, st.session_state.independent_vars,
     st.session_state.dependent_vars, st.session_state.variable_stats,
     _, st.session_state.unique_variable_values,
     st.session_state.variable_descriptions,
     st.session_state.detected_binary_vars) = analyze_csv(df)
    st.session_state.processed_file = project_name

    # Automatically run the analysis for ALL dependent variables
    if not st.session_state.get('analysis_done'):
        df_for_analysis = st.session_state.exp_df.copy()
        expand_terms(df_for_analysis, st.session_state.independent_vars)
        st.session_state.expanded_df = df_for_analysis
        
        current_wrapped_models = {}
        for dep_var in st.session_state.dependent_vars:
            wrapped_model = run_analysis(
                df_for_analysis, st.session_state.independent_vars, 
                dep_var, 'Polynomial OLS', {}
            )
            current_wrapped_models[dep_var] = wrapped_model

        st.session_state.wrapped_models = current_wrapped_models
        st.session_state.analysis_done = True
        st.session_state.active_analysis_run = "預設 OLS 分析"

def main():
    """
    執行互動式儀表板教育版 Streamlit 應用程式的主函數。
    """
    st.set_page_config(
        page_title="AIPRS 互動儀表板 (教育版)",
        page_icon="🔬",
        layout="wide"
    )

    # 初始化 session state
    state_management.initialize_session_state()

    # 自動載入專案和分析
    auto_load_and_analyze_project()

    # 渲染儀表板的主體內容
    dashboard_view.render()


if __name__ == "__main__":
    main()