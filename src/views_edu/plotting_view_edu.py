# src/views_edu/plotting_view_edu.py
import streamlit as st
from utils.ui_helpers import format_variable_options, display_surface_plot

def render():
    """渲染圖表工具介面 (教育版) 的所有 UI 元件與邏輯。"""
    st.header("圖表視覺化")
    st.info("透過 3D 反應曲面圖，我們可以直觀地看到兩個主要變因如何共同影響最終的結果。")

    formatted_models = format_variable_options(st.session_state.wrapped_models.keys())

    # --- 圖表設定 ---
    with st.container(border=True):
        st.subheader("圖表設定")
        col1, col2, col3 = st.columns(3)

        # 選擇模型
        selected_model_formatted = col1.selectbox("選擇您想分析的結果 (Z 軸)", options=formatted_models, key="edu_plot_model")
        model_to_plot = selected_model_formatted.split(":")[0]

        # 選擇變因
        formatted_vars = format_variable_options(st.session_state.independent_vars)
        selected_x_formatted = col2.selectbox("選擇主要變因 (X 軸)", options=formatted_vars, key="edu_plot_x")
        x_var = selected_x_formatted.split(":")[0]

        y_options_formatted = [v for v in formatted_vars if not v.startswith(x_var)]
        selected_y_formatted = col3.selectbox("選擇次要變因 (Y 軸)", options=y_options_formatted, key="edu_plot_y")
        y_var = selected_y_formatted.split(":")[0]

        # 固定其他變數
        fixed_vars = {}
        other_vars = [v for v in st.session_state.independent_vars if v not in [x_var, y_var]]
        if other_vars:
            st.markdown("**固定其他變數的值：**")
            for var in other_vars:
                # 簡單起見，直接使用滑桿讓使用者選擇
                min_val, _, max_val = st.session_state.variable_stats[var]
                desc = st.session_state.variable_descriptions.get(var, var)
                fixed_vars[var] = st.slider(
                    f"固定變數：{desc}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2), # 預設為中間值
                    key=f"edu_plot_fixed_{var}"
                )

    # --- 繪製圖表 ---
    st.subheader("3D 反應曲面圖")
    
    # 準備繪圖參數，許多參數使用預設值以簡化介面
    plot_parameters = {
        'x_var': x_var,
        'y_var': y_var,
        'z_var_1': model_to_plot,
        'fixed_vars_dict_1': fixed_vars,
        'variable_descriptions': st.session_state.variable_descriptions,
        'show_actual_data': True, # 預設顯示實際數據點
        'colorscale_1': 'Viridis', # 預設色盤
    }
    
    # 直接呼叫與專業版相同的繪圖函式
    display_surface_plot(plot_parameters)