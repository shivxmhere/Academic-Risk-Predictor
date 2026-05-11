import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def highlight_rows(row):
    val = row['Predicted Risk']
    if val == 'High Risk':
        return ['background-color: #FEE2E2; color: #000000; font-weight: 500;'] * len(row)
    elif val == 'Medium Risk':
        return ['background-color: #FEF9C3; color: #000000; font-weight: 500;'] * len(row)
    elif val == 'Low Risk':
        return ['background-color: #DCFCE7; color: #000000; font-weight: 500;'] * len(row)
    return [''] * len(row)

def show_predict_page():
    st.title("🔮 Predict Academic Risk")
    
    # 1. Check if raw_data exists
    if 'raw_data' not in st.session_state or st.session_state['raw_data'] is None:
        st.warning("No data found. Please go to the 'Upload Data' page and upload a student dataset first.")
        return
        
    try:
        # 2. Feature engineering
        df = st.session_state['raw_data'].copy()
        df['api'] = (df['attendance_pct'] * 0.3) + (df['internal_score'] * 0.4) + (df['prev_grade'] * 0.3)
        
        # 3. Scale using saved scaler
        features = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours', 'api']
        scaler = st.session_state['scaler']
        X_scaled = scaler.transform(df[features])
        
        # 4. Predict
        model = st.session_state['model']
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        risk_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        df['Predicted Risk'] = [risk_map[p] for p in predictions]
        df['Confidence %'] = (np.max(probabilities, axis=1) * 100).round(1)
        
        # 8. Store results for export
        st.session_state['results'] = df.copy()
        
        # 5. Summary metric row
        st.subheader("Dashboard Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        high_risk_count = (df['Predicted Risk'] == 'High Risk').sum()
        medium_risk_count = (df['Predicted Risk'] == 'Medium Risk').sum()
        low_risk_count = (df['Predicted Risk'] == 'Low Risk').sum()
        total_count = len(df)
        
        col1.metric("Low Risk 🟢", low_risk_count)
        col2.metric("Medium Risk 🟡", medium_risk_count)
        col3.metric("High Risk 🔴", high_risk_count)
        col4.metric("Total Students", total_count)
        
        # 6. Colour-coded results table
        st.subheader("Predictions Table")
        styled_df = df.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # 7. Individual student profile section
        st.markdown("---")
        st.subheader("Individual Student Profile")
        
        if 'student_id' in df.columns:
            student_list = df['student_id'].tolist()
        else:
            student_list = df.index.tolist()
            
        selected_student = st.selectbox("Select a student to view details:", student_list)
        
        if selected_student is not None:
            if 'student_id' in df.columns:
                student_data = df[df['student_id'] == selected_student].iloc[0]
            else:
                student_data = df.loc[selected_student]
                
            risk = student_data['Predicted Risk']
            conf = student_data['Confidence %']
            
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.write("**Feature Values:**")
                fig, ax = plt.subplots(figsize=(8, 4))
                # Only plot the original 6 features (without api for cleaner visualization)
                plot_features = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours']
                values = student_data[plot_features].values
                ax.barh(plot_features, values, color='skyblue')
                ax.set_xlabel('Value')
                plt.tight_layout()
                st.pyplot(fig)
                
            with c2:
                # Badge rendering with exact text requested
                if risk == 'High Risk':
                    badge_color = "#FEE2E2"
                    text_color = "#991B1B"
                    rec = "Immediately contact the student, alert their faculty advisor, and schedule a counselling session."
                elif risk == 'Medium Risk':
                    badge_color = "#FEF9C3"
                    text_color = "#854D0E"
                    rec = "Schedule a one-on-one check-in within the next two weeks."
                else:
                    badge_color = "#DCFCE7"
                    text_color = "#166534"
                    rec = "No immediate action required. Continue monitoring."
                
                html_badge = f"""
                <div style="background-color: {badge_color}; color: {text_color}; padding: 20px; border-radius: 10px; margin-top: 10px; border: 1px solid {text_color};">
                    <h2 style="margin:0; color: {text_color};">{risk}</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px; margin-bottom: 5px;">Confidence: <b>{conf}%</b></p>
                    <hr style="border-top: 1px solid {text_color}; opacity: 0.3; margin: 15px 0;">
                    <p style="margin:0;"><b>Recommended Action:</b><br>{rec}</p>
                </div>
                """
                st.markdown(html_badge, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
