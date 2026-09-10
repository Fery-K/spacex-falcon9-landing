import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SpaceX Falcon 9 - Booster Recovery",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_pipeline_and_features():
    pipeline = joblib.load("./model/model_logreg.joblib"")
    if hasattr(pipeline, 'best_estimator_'):
        pipeline = pipeline.best_estimator_
        
    feature_cols = joblib.load("./model/model_features.joblib")
    return pipeline, feature_cols

try:
    model, expected_features = load_pipeline_and_features()
except Exception:
    model, expected_features = None, None

# --- SIDEBAR (AUTHOR CREDITS & INFO) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/de/SpaceX-Logo.svg", width=200)
    st.markdown("### SpaceX Booster Landing Recovery Prediction")
    st.markdown("Operational deployment for model demo purposes in evaluating first-stage booster landing recovery using **machine learning**.")
    st.markdown("<br> <br> <br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("IBM Applied Data Science Capstone Project")
    st.markdown("""
    <div style="
        font-size: 11px; 
        color: #888888; 
        line-height: 1.4; 
        border-top: 1px solid #333333; 
        padding-top: 10px; 
        margin-top: 1px;">
        <strong>Disclaimer:</strong> This application is an independent student capstone 
        project developed solely for academic and demonstration purposes as part of the 
        <strong>IBM Applied Data Science Professional Certificate</strong>. <br>
        It is not affiliated with, authorized, sponsored, or officially endorsed by 
        <strong>Space Exploration Technologies Corp. (SpaceX)</strong>, IBM, or any of 
        their subsidiaries. All trademarks, service marks, and company names are the 
        property of their respective owners. Predictions are generated via an 
        learning model and do not reflect real-time launch operations or flight guarantees.
    </div>
    """, unsafe_allow_html=True)

# --- MAIN SCREENING PAGE ---
st.title("Rocket Launch First-Stage Booster Landing Prediction")

tab1 = st.tabs(["Launch Screening"])
with tab1[0]:
    with st.form("Mission_Input_Form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Mission Profile")
            flight_number = st.number_input(
                "Flight Number", 
                min_value=1, 
                max_value=250, 
                value=91, 
                step=1,
                help="Sequential launch sequence number indicating SpaceX's cumulative operational experience."
            )
            payload_mass = st.number_input(
                "Payload Mass (kg)", 
                min_value=0.0, 
                max_value=16000.0, 
                value=6000.0, 
                step=100.0,
                help="Total net payload mass in kilograms delivered to the destination orbital trajectory."
            )
            orbit = st.selectbox(
                "Target Orbit",
                ['ES-L1', 'GEO', 'GTO', 'HEO', 'ISS', 'LEO', 'MEO', 'PO', 'SO', 'SSO', 'VLEO'],
                index=5,
                help="Target destination orbit."
            )

        with col2:
            st.subheader("2. Launch & Recovery Site")
            launch_site = st.selectbox(
                "Launch Complex Site",
                ['CCAFS SLC 40', 'KSC LC 39A', 'VAFB SLC 4E'],
                index=0,
                help="Ground launch complex facility from which the rocket lifts off."
            )
            pad = st.selectbox(
                "Landing Pad",
                ['LZ-1', 'OCISLY', 'JRTI', 'LZ-2', 'Not Expandable'],
                index=0,
                help="Designated recovery platform."
            )
            landing_pad = {
                'LZ-1': '5e9e3032383ecb267a34e7c7', 
                'OCISLY':'5e9e3032383ecb554034e7c9', 
                'JRTI':'5e9e3032383ecb6bb234e7ca', 
                'LZ-2':'5e9e3032383ecb761634e7cb', 
                'Not Expandable':'None'
            }[pad]
            grid_fins = st.radio(
                "Grid Fins Installed?", 
                ["Yes", "No"], 
                horizontal=True,
                help="Steerable aerodynamic control surfaces used during atmospheric descent."
            )

        with col3:
            st.subheader("3. Booster Hardware")
            block = st.selectbox(
                "Booster Block Iteration", 
                [1, 2, 3, 4, 5], 
                index=4,
                help="Hardware revision tier (Block 5 is the modern, highly reusable standard)."
            )
            flights = st.slider(
                "Total Flights Flown by this Core", 
                min_value=1, 
                max_value=20, 
                value=1,
                help="Total cumulative launches logged by this specific booster core."
            )
            reused_count = st.slider(
                "Historical Reused Times", 
                min_value=0, 
                max_value=10, 
                value=0,
                help="Historical count of prior recovery missions completed by this hardware."
            )
            
            legs = st.radio(
                "Landing Legs Equipped?", 
                ["Yes", "No"], 
                horizontal=True,
                help="Deployable legs required for soft surface touchdown."
            )
            reused = "Yes" if flights > 1 else "No"

        submitted = st.form_submit_button("Predict Landing Outcome", use_container_width=True)

    if submitted:
        if model is None or expected_features is None:
            st.error("Model asset not found in `./model/`. Please verify `model_logreg.joblib` and `model_features.joblib`.")
        else:
            with st.spinner("Processing launch parameters through Model Pipeline..."):
                time.sleep(1)

            # 1. Map raw input dictionary
            input_data = pd.DataFrame([{
            'FlightNumber': flight_number,
            'PayloadMass': payload_mass,
            'Flights': flights,
            'GridFins': int(grid_fins == "Yes"),
            'Reused': int(reused == "Yes"),
            'Legs': int(legs == "Yes"),
            'Block': block,
            'ReusedCount': reused_count,
            'Orbit': orbit,
            'LaunchSite': launch_site,
            'LandingPad': 'None_Expendable' if landing_pad == 'None' else landing_pad
            }])

            # 2. Pipeline inference
            probabilities = model.predict_proba(input_data)[0]
            prob_fail = probabilities[0]
            prob_success = probabilities[1]
            
            prediction = model.predict(input_data)[0]
            decision_label = "SUCCESS" if prediction == 1 else "FAIL"

            # Store result in session_state for Tab 2
            st.session_state['pred_done'] = True
            st.session_state['prob_success'] = prob_success
            st.session_state['prob_fail'] = prob_fail
            st.session_state['flight_number'] = flight_number

            st.divider()

            # --- VISUALIZATION SECTION ---
            st.subheader("Prediction Result")

            label_df = pd.DataFrame({
                'label': ['First Stage Recovery'],
                'Success': [prob_success],
                'Failure': [prob_fail]
            })

            base = alt.Chart(label_df)
            middle = base.encode(
                y=alt.Y('label:N', axis=None),
                text=alt.Text('label:N'),
                color=alt.value('white')
            ).mark_text(fontWeight='bold').properties(width=160)

            # Left bar (Failure / Red)
            left = base.encode(
                y=alt.Y('label:N', axis=None),
                x=alt.X('Failure:Q', title='', scale=alt.Scale(domain=[0, 1]), sort=alt.SortOrder('descending')).axis(format='%'),
                color=alt.value('#E64242')
            ).mark_bar().properties(title=alt.TitleParams(text='Fail Landing', dx=340))

            # Right bar (Success / Green)
            right = base.encode(
                y=alt.Y('label:N', axis=None),
                x=alt.X('Success:Q', title='', scale=alt.Scale(domain=[0, 1])).axis(format='%'),
                color=alt.value('#2ECC71')
            ).mark_bar().properties(title='Success Landing')

            pred_chart = alt.hconcat(left, middle, right, spacing=5)
            st.altair_chart(pred_chart, use_container_width=True)

            col_res1, col_res2 = st.columns([1, 2], gap="large")
    
            with col_res1:
                st.subheader("Classification Outcome")
                
                if prob_success >= model.threshold:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #d4edda; 
                            color: #155724; 
                            padding: 15px; 
                            border-radius: 5px; 
                            border-left: 5px solid #28a745;
                            margin-bottom: 10px;">
                            <p style="margin:0; font-size:14px; text-transform: uppercase; font-weight: bold; opacity: 0.8;">Class Probabilities</p>
                            <p style="margin:0; font-size:32px; font-weight: bold;">{prob_success * 100:.1f}%</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f8d7da; 
                            color: #721c24; 
                            padding: 15px; 
                            border-radius: 5px; 
                            border-left: 5px solid #dc3545;
                            margin-bottom: 10px;">
                            <p style="margin:0; font-size:14px; text-transform: uppercase; font-weight: bold; opacity: 0.8;">Outcome</p>
                            <p style="margin:0; font-size:24px; font-weight: bold;">{decision_label}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
    
                st.caption(f"Risk of booster failure: {prob_fail * 100:.1f}%")
                
                st.info(f"""
                * **Launch Site:** `{launch_site}`
                * **Target Orbit:** `{orbit}`
                * **Core Iteration:** Block {block} (Flight {flights})
                """)
    
            with col_res2:
                z_val = np.log((prob_success + 1e-9) / (1.0 - prob_success + 1e-9))
                z_range = np.linspace(-6, 6, 200)
                sigmoid_df = pd.DataFrame({
                    'LogOdds': z_range,
                    'Probability': 1 / (1 + np.exp(-z_range))
                })
                point_df = pd.DataFrame({'LogOdds': [z_val], 'Probability': [prob_success]})
    
                sigmoid_curve = alt.Chart(sigmoid_df).mark_line(color='#4A90E2', strokeWidth=2.5).encode(
                    x=alt.X('LogOdds:Q', title='Decision Boundary Margin (w^T * X + b)'),
                    y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%', title='P(Success)'))
                )
    
                threshold_line = alt.Chart(pd.DataFrame({'y': [model.threshold]})).mark_rule(strokeDash=[4, 4], color='gray').encode(y='y:Q')
    
                point_marker = alt.Chart(point_df).mark_point(
                    size=160, color='#E64242' if prob_success < 0.5 else '#2ECC71', filled=True
                ).encode(
                    x='LogOdds:Q',
                    y='Probability:Q',
                    tooltip=[alt.Tooltip('Probability:Q', format='.2%'), alt.Tooltip('LogOdds:Q', format='.2f')]
                )
    
                chart_logreg = (sigmoid_curve + threshold_line + point_marker).properties(
                    height=370,
                    title=f""
                )
                st.altair_chart(chart_logreg, use_container_width=True)
