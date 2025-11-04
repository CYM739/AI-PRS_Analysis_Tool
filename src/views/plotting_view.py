# src/views/plotting_view.py
import streamlit as st
from logic.plotting import plot_tradeoff_contour, plot_tradeoff_analysis, plot_response_curve
from utils.ui_helpers import format_variable_options, display_surface_plot

def render():
    """Renders all UI components and logic for the Plotting Tools tab."""
    st.subheader("Plotting Tools")

    if not st.session_state.get('wrapped_models'):
        st.info("Run an analysis from the 'Project Library' to generate models and view plots.")
        return
        
    formatted_models = format_variable_options(st.session_state.wrapped_models.keys())

    if len(st.session_state.independent_vars) >= 2:
        plot_type = st.radio(
            "Select Plot Type",
            ["3D Surface", "2D Trade-Off Contour", "2D Trade-Off Analysis"],
            horizontal=True,
            captions=["", "", "Visualize how two models change relative to one variable."]
        )
        st.write("---")

        if plot_type == "3D Surface":
            st.subheader("Generate a 3D Response Surface Plot")
            col1, col2, col3 = st.columns(3)

            selected_model_formatted_1 = col1.selectbox("Select Primary Model", options=formatted_models, key="main_plot_model_1")
            model_to_plot_1 = selected_model_formatted_1.split(":")[0]

            formatted_vars = format_variable_options(st.session_state.independent_vars)
            selected_x_formatted = col2.selectbox("X-axis variable", options=formatted_vars, key="main_plot_x")
            x_var = selected_x_formatted.split(":")[0]

            y_options_formatted = [v for v in formatted_vars if not v.startswith(x_var)]
            selected_y_formatted = col3.selectbox("Y-axis variable", options=y_options_formatted, key="main_plot_y")
            y_var = selected_y_formatted.split(":")[0]

            show_actual_data_points = st.toggle("Show actual data points on the plot", value=True, key="main_plot_toggle_data")

            fixed_vars_1 = {}
            other_vars = [v for v in st.session_state.independent_vars if v not in [x_var, y_var]]

            if other_vars:
                st.write("---")
                st.write("**Set Fixed Values for Other Variables:**")
                for var in other_vars:
                    unique_vals = st.session_state.unique_variable_values.get(var, [])
                    _ , second_min, _ = st.session_state.variable_stats[var]
                    default_index = 0
                    if unique_vals and second_min in unique_vals:
                        try:
                            default_index = unique_vals.index(second_min)
                        except ValueError:
                            default_index = 0

                    desc = st.session_state.variable_descriptions.get(var, 'No description')
                    fixed_vars_1[var] = st.selectbox(
                        label=f"Fixed value for {var} ({desc})",
                        options=unique_vals,
                        index=default_index,
                        key=f"main_plot_select_{var}_1"
                    )
            st.write("---")

            model_to_plot_2 = None
            colorscale_2 = 'Greys'
            colorscale_options = ['Viridis', 'Plasma', 'Jet', 'Cividis', 'Hot', 'Cool', 'Blues', 'Greens', 'RdBu', 'Greys']

            with st.expander("📊 Compare with another model"):
                compare_models = st.checkbox("Overlay a second model surface")
                if compare_models:
                    model_options_2 = [m for m in formatted_models if not m.startswith(model_to_plot_1)]
                    if not model_options_2:
                        st.warning("Only one model is available. Cannot perform comparison.")
                    else:
                        c1_comp, c2_comp = st.columns([2,1])
                        with c1_comp:
                            selected_model_formatted_2 = st.selectbox(
                                "Select Comparison Model",
                                options=model_options_2,
                                key="main_plot_model_2"
                            )
                            model_to_plot_2 = selected_model_formatted_2.split(":")[0]
                        with c2_comp:
                            colorscale_2 = st.selectbox("Comparison Color Scheme", options=colorscale_options, key="colorscale_2", index=9)

                        st.info("The comparison surface will be plotted using the same fixed values as the primary model to ensure a valid comparison.")

            with st.expander("🎨 Customize Plot Appearance"):
                st.write("**Titles and Labels**")
                custom_main_title = st.text_input("Main Title", placeholder="Leave blank for default")
                ax_c1, ax_c2, ax_c3 = st.columns(3)
                custom_x_title = ax_c1.text_input("X-axis Title", placeholder="Leave blank for default")
                custom_y_title = ax_c2.text_input("Y-axis Title", placeholder="Leave blank for default")
                custom_z_title = ax_c3.text_input("Z-axis Title", placeholder="Leave blank for default")
                axis_title_font_size = st.slider("Axis Title Font Size", min_value=8, max_value=24, value=12, key="main_plot_3d_axis_font")
                axis_tick_font_size = st.slider("Axis Tick Label Font Size (Numbers)", min_value=8, max_value=24, value=10, key="main_plot_3d_axis_tick_font")
                
                st.write("**Z-Axis Range**")
                z_c1, z_c2 = st.columns(2)
                z_min = z_c1.number_input("Min Z Value", value=float(st.session_state.exp_df[model_to_plot_1].min()), key="plot_z_min")
                z_max = z_c2.number_input("Max Z Value", value=float(st.session_state.exp_df[model_to_plot_1].max()), key="plot_z_max")
                z_range = [z_min, z_max]
                
                st.write("---")
                st.write("**Colors and Grids**")
                cr_c1, cr_c2 = st.columns(2)
                custom_colorscale_1 = cr_c1.selectbox("Primary Surface Color Scheme", options=colorscale_options)
                with cr_c2:
                    st.write("Surface Grid Lines")
                    show_x_grid = st.checkbox("Show X-axis grid", value=False)
                    show_y_grid = st.checkbox("Show Y-axis grid", value=False)

                st.write("---")
                st.write("**Export Settings**")
                download_scale = st.number_input("Download Resolution Scale", min_value=1.0, max_value=10.0, value=1.0, step=0.5,help="Increase for higher resolution PNG downloads.")

            plot_parameters = {
                'x_var': x_var,
                'y_var': y_var,
                'z_var_1': model_to_plot_1,
                'fixed_vars_dict_1': fixed_vars_1,
                'z_var_2': model_to_plot_2,
                'fixed_vars_dict_2': fixed_vars_1 if model_to_plot_2 else None,
                'variable_descriptions': st.session_state.variable_descriptions,
                'show_actual_data': show_actual_data_points,
                'colorscale_1': custom_colorscale_1,
                'colorscale_2': colorscale_2,
                'main_title': custom_main_title or None,
                'x_title': custom_x_title or None,
                'y_title': custom_y_title or None,
                'z_title': custom_z_title or None,
                'z_range': z_range,
                'show_x_grid': show_x_grid,
                'show_y_grid': show_y_grid,
                'axis_title_font_size': axis_title_font_size,
                'axis_tick_font_size': axis_tick_font_size,
                'download_scale': download_scale
            }
            plot_config = {'toImageButtonOptions': {'format': 'png','filename': f'{model_to_plot_1}_surface_plot','height': 700,'width': 700,'scale': download_scale}}
            display_surface_plot(plot_parameters, plot_config)

        # ✅ BUG FIX: Corrected the indentation of this entire 'elif' block
        elif plot_type == "2D Trade-Off Contour":
            st.subheader("Generate a 2D Trade-Off Contour Plot")
            if len(formatted_models) < 2:
                st.warning("You need at least two outcome models to generate a trade-off plot.")
            else:
                c1, c2 = st.columns(2)
                model_1_formatted = c1.selectbox("Select Model 1 (Red regions)", options=formatted_models, key="contour_model_1")
                model_1_name = model_1_formatted.split(":")[0]

                model_2_options = [m for m in formatted_models if not m.startswith(model_1_name)]
                model_2_formatted = c2.selectbox("Select Model 2 (Blue regions)", options=model_2_options, key="contour_model_2")
                model_2_name = model_2_formatted.split(":")[0]

                c3, c4 = st.columns(2)
                formatted_vars = format_variable_options(st.session_state.independent_vars)
                x_var_formatted = c3.selectbox("X-axis variable", options=formatted_vars, key="contour_x")
                x_var_contour = x_var_formatted.split(":")[0]

                y_options_formatted = [v for v in formatted_vars if not v.startswith(x_var_contour)]
                y_var_formatted = c4.selectbox("Y-axis variable", options=y_options_formatted, key="contour_y")
                y_var_contour = y_var_formatted.split(":")[0]

                fixed_vars_contour = {}
                other_vars_contour = [v for v in st.session_state.independent_vars if v not in [x_var_contour, y_var_contour]]

                if other_vars_contour:
                    st.write("---")
                    st.write("**Set Fixed Values for Other Variables:**")
                    for var in other_vars_contour:
                        unique_vals = st.session_state.unique_variable_values.get(var, [])
                        _, second_min, _ = st.session_state.variable_stats[var]
                        default_index = 0
                        if unique_vals and second_min in unique_vals:
                            try: default_index = unique_vals.index(second_min)
                            except ValueError: default_index = 0
                        desc = st.session_state.variable_descriptions.get(var, 'No description')
                        fixed_vars_contour[var] = st.selectbox(
                            label=f"Fixed value for {var} ({desc})", options=unique_vals,
                            index=default_index, key=f"contour_select_{var}"
                        )

                if st.button("Generate 2D Trade-Off Plot"):
                    model_1_obj = st.session_state.wrapped_models[model_1_name]
                    model_2_obj = st.session_state.wrapped_models[model_2_name]
                    fig = plot_tradeoff_contour(
                        OLS_model_1=model_1_obj, OLS_model_2=model_2_obj, model_name_1=model_1_name,
                        model_name_2=model_2_name, all_alphabet_vars=st.session_state.independent_vars,
                        x_var=x_var_contour, y_var=y_var_contour, fixed_vars_dict=fixed_vars_contour,
                        dataframe=st.session_state.expanded_df, variable_descriptions=st.session_state.variable_descriptions
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ✅ BUG FIX: Corrected the indentation of this entire 'elif' block
        elif plot_type == "2D Trade-Off Analysis":
            st.subheader("Generate a 2D Trade-Off Analysis Plot")
            st.info("Select two models and an independent variable to see how the outcomes and their difference change as that single variable changes.")

            if len(st.session_state.wrapped_models.keys()) >= 2:
                v_col1, v_col2, v_col3 = st.columns(3)
                model_1_formatted = v_col1.selectbox("Select Model 1", options=formatted_models, key="tradeoff_model_1")
                model_1_name = model_1_formatted.split(":")[0]
                model_2_options = [m for m in formatted_models if not m.startswith(model_1_name)]
                model_2_formatted = v_col2.selectbox("Select Model 2", options=model_2_options, key="tradeoff_model_2")
                model_2_name = model_2_formatted.split(":")[0]
                formatted_ind_vars = format_variable_options(st.session_state.independent_vars)
                x_var_tradeoff_formatted = v_col3.selectbox("Variable to Change (X-axis)", options=formatted_ind_vars, key="tradeoff_x_var")
                x_var_tradeoff = x_var_tradeoff_formatted.split(":")[0]
                fixed_vars_tradeoff = {}
                other_vars_tradeoff = [v for v in st.session_state.independent_vars if v != x_var_tradeoff]

                if other_vars_tradeoff:
                    st.write("**Set Fixed Values for Other Variables:**")
                    for var in other_vars_tradeoff:
                        unique_vals = st.session_state.unique_variable_values.get(var, [])
                        _, second_min, _ = st.session_state.variable_stats[var]
                        default_index = 0
                        if unique_vals and second_min in unique_vals:
                            try: default_index = unique_vals.index(second_min)
                            except ValueError: default_index = 0
                        desc = st.session_state.variable_descriptions.get(var, 'No description')
                        fixed_vars_tradeoff[var] = st.selectbox(
                            label=f"Fixed value for {var} ({desc})", options=unique_vals,
                            index=default_index, key=f"tradeoff_select_{var}"
                        )

                if st.button("Generate Trade-Off Plot"):
                    model_1_obj = st.session_state.wrapped_models[model_1_name]
                    model_2_obj = st.session_state.wrapped_models[model_2_name]
                    fig = plot_tradeoff_analysis(
                        model_1=model_1_obj, model_2=model_2_obj, model_name_1=model_1_name,
                        model_name_2=model_2_name, all_alphabet_vars=st.session_state.independent_vars,
                        x_var=x_var_tradeoff, fixed_vars_dict=fixed_vars_tradeoff,
                        dataframe=st.session_state.expanded_df, variable_descriptions=st.session_state.variable_descriptions
                    )
                    st.pyplot(fig)
            else:
                st.warning("You need at least two outcome models to generate a trade-off plot.")

    # ✅ BUG FIX: Corrected the indentation of this entire 'else' block
    else: 
        st.subheader("Generate 2D Response Curve")
        st.info("Since there is only one independent variable, an interactive 2D plot is shown to illustrate its effect on the outcome.")
        selected_model_formatted = st.selectbox("Select Model to Plot", options=formatted_models, key="2d_plot_model")
        model_to_plot = selected_model_formatted.split(":")[0]
        single_var = st.session_state.independent_vars[0]
        model_obj = st.session_state.wrapped_models[model_to_plot]

        with st.expander("🎨 Customize Plot Appearance"):
            show_actual_data_2d = st.toggle("Show actual data points", value=True, key="main_plot_2d_toggle_data")
            st.write("**Titles and Labels**")
            custom_main_title_2d = st.text_input("Main Title", placeholder="Leave blank for default", key="main_plot_2d_title")
            ax_c1, ax_c2 = st.columns(2)
            custom_x_title_2d = ax_c1.text_input("X-axis Title", placeholder="Leave blank for default", key="main_plot_2d_xtitle")
            custom_y_title_2d = ax_c2.text_input("Y-axis Title", placeholder="Leave blank for default", key="main_plot_2d_ytitle")
            axis_font_size_2d = st.slider("Axis Title Font Size", 8, 24, 12, key="main_plot_2d_axis_font")
            st.write("---")
            st.write("**Colors and Sizes**")
            c1, c2 = st.columns(2)
            actual_color_2d = c1.color_picker("Actual Data Color", value="#FF4B4B", key="main_plot_2d_actual_color")
            curve_color_2d = c2.color_picker("Model Curve Color", value="#0068C9", key="main_plot_2d_curve_color")
            c3, c4 = st.columns(2)
            actual_size_2d = c3.slider("Actual Data Size", 4, 20, 8, key="main_plot_2d_actual_size")
            curve_width_2d = c4.slider("Model Curve Width", 1, 10, 3, key="main_plot_2d_curve_width")

        fig = plot_response_curve(
            dataframe=st.session_state.expanded_df, OLS_model=model_obj,
            independent_var=single_var, dependent_var=model_to_plot,
            variable_descriptions=st.session_state.variable_descriptions,
            main_title=custom_main_title_2d or None, x_title=custom_x_title_2d or None,
            y_title=custom_y_title_2d or None, axis_title_font_size=axis_font_size_2d,
            actual_color=actual_color_2d, curve_color=curve_color_2d,
            actual_size=actual_size_2d, curve_width=curve_width_2d, show_actual_data=show_actual_data_2d
        )
        st.plotly_chart(fig, use_container_width=True)
