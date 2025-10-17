# src/views_edu/library_view_edu.py
import streamlit as st
import pandas as pd
import pickle
import base64
from logic.data_processing import analyze_csv, run_analysis, expand_terms
from utils.state_management import reset_state, initialize_session_state
from views.library_view import load_library, save_library # 直接沿用專業版的函式

def render():
    """渲染專案資料庫介面 (教育版) 的所有 UI 元件與邏輯。"""
    st.header("專案資料庫與模型分析")
    st.info("歡迎使用 AIPRS 教育版！請依照以下步驟來完成您的分析。")

    # --- 步驟一：建立新專案 ---
    with st.container(border=True):
        st.subheader("第一步：上傳資料並建立專案")
        st.markdown("請上傳您的 CSV 格式實驗數據。檔案中應包含數個代表**變因**的欄位，以及一個或多個代表**結果**的欄位。")

        uploaded_file = st.file_uploader("點擊此處上傳您的 CSV 檔案", type=["csv"])
        project_name_input = st.text_input("請為您的專案命名", "我的教學分析專案")

        if st.button("建立新專案", type="primary"):
            if uploaded_file is not None and project_name_input:
                library = load_library()
                if project_name_input in library:
                    st.error(f"名為 '{project_name_input}' 的專案已存在，請換一個名稱。")
                else:
                    try:
                        # 優先嘗試 UTF-8
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        # 若失敗，則嘗試 Big5 (ms950)
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='ms950')

                    (data_df, _, independent_vars, dependent_vars,
                     _, _, _, _, _) = analyze_csv(df)

                    if not dependent_vars or not independent_vars:
                        st.error("資料驗證失敗：請確認您的 CSV 檔案中包含符合命名規則的變因與結果欄位。")
                        st.stop()
                    
                    df_json = df.to_json(orient='split')
                    library[project_name_input] = {
                        "data_df_json": df_json,
                        "analysis_runs": {}
                    }
                    save_library(library)
                    st.success(f"成功建立專案 '{project_name_input}'！已自動為您載入資料。")
                    
                    reset_state()
                    (st.session_state.exp_df, st.session_state.all_vars, st.session_state.independent_vars,
                     st.session_state.dependent_vars, st.session_state.variable_stats,
                     _, st.session_state.unique_variable_values,
                     st.session_state.variable_descriptions,
                     st.session_state.detected_binary_vars) = analyze_csv(df)
                    st.session_state.processed_file = project_name_input
                    st.rerun()
            else:
                st.warning("請務必上傳檔案並為專案命名。")

    st.divider()

    # --- 新增的專案管理區塊 ---
    library = load_library()
    if library: # 只有當資料庫中有專案時才顯示
        with st.container(border=True):
            st.subheader("專案管理 (載入與刪除)")
            
            projects = sorted(list(library.keys()))
            selected_project = st.selectbox("選擇一個已存在的專案", projects)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("載入選定專案", use_container_width=True):
                    project_data = library[selected_project]
                    df_json = project_data['data_df_json']
                    df = pd.read_json(df_json, orient='split')

                    reset_state()
                    (st.session_state.exp_df, st.session_state.all_vars, st.session_state.independent_vars,
                     st.session_state.dependent_vars, st.session_state.variable_stats,
                     _, st.session_state.unique_variable_values,
                     st.session_state.variable_descriptions,
                     st.session_state.detected_binary_vars) = analyze_csv(df)
                    st.session_state.processed_file = selected_project
                    st.success(f"已成功載入專案 '{selected_project}'。")
                    st.rerun()
            
            with col2:
                if st.button("刪除選定專案", type="secondary", use_container_width=True):
                    if selected_project in library:
                        del library[selected_project]
                        save_library(library)
                    if st.session_state.get('processed_file') == selected_project:
                        reset_state()
                    st.success(f"已刪除專案 '{selected_project}'")
                    st.rerun()

    # --- 步驟二：執行分析 ---
    if st.session_state.get('processed_file'):
        with st.container(border=True):
            st.subheader(f"第二步：為專案「{st.session_state.processed_file}」執行模型分析")
            st.info("教育版預設使用「多項式 OLS (Ordinary Least Squares)」模型。這是一種統計學方法，用於建立變因與結果之間的數學方程式，是反應曲面分析的基礎。")

            analysis_name = "預設 OLS 分析"
            st.markdown(f"分析名稱將設定為： **{analysis_name}**")

            if st.button("🚀 開始執行分析", type="primary", use_container_width=True):
                project_data = library[st.session_state.processed_file]
                if analysis_name in project_data.get("analysis_runs", {}):
                    st.warning(f"名為 '{analysis_name}' 的分析已存在。結果將被覆蓋。")

                with st.spinner("正在建立數學模型，請稍候..."):
                    try:
                        df_for_analysis = st.session_state.exp_df.copy()
                        expand_terms(df_for_analysis, st.session_state.independent_vars)
                        st.session_state.expanded_df = df_for_analysis
                        
                        models_to_store = {}
                        current_wrapped_models = {}

                        for dep_var in st.session_state.dependent_vars:
                            clean_df = df_for_analysis.dropna(subset=[dep_var] + st.session_state.independent_vars).copy()
                            wrapped_model = run_analysis(
                                clean_df, st.session_state.independent_vars, dep_var, 'Polynomial OLS', {}
                            )
                            y_actual = clean_df[dep_var]
                            y_predicted = wrapped_model.predict(clean_df)
                            avp_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})
                            
                            current_wrapped_models[dep_var] = wrapped_model
                            pickled_model = pickle.dumps(wrapped_model)
                            b64_model = base64.b64encode(pickled_model).decode('utf-8')
                            
                            models_to_store[dep_var] = {
                                "model_wrapper_b64": b64_model,
                                "summary_text": wrapped_model.get_summary(),
                                "actual_vs_predicted_csv": avp_df.to_csv(index=False)
                            }
                        
                        library[st.session_state.processed_file]['analysis_runs'][analysis_name] = {
                            "model_type": 'Polynomial OLS',
                            "models": models_to_store
                        }
                        save_library(library)
                        
                        st.session_state.wrapped_models = current_wrapped_models
                        st.session_state.analysis_done = True
                        st.session_state.active_analysis_run = analysis_name
                        st.success("模型分析完成！您現在可以前往「圖表視覺化」與「AI 智慧優化」分頁探索結果。")
                        st.rerun()

                    except Exception as e:
                        st.error(f"分析過程中發生錯誤: {e}")