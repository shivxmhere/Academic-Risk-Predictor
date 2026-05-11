import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_analytics_page():
    st.title("📊 Analytics Dashboard")
    
    if 'results' not in st.session_state or st.session_state['results'] is None:
        st.warning("No prediction results found. Please go to the 'Predict Risk' page and run predictions first.")
        return
        
    df = st.session_state['results']
    model = st.session_state.get('model')
    
    color_map = {'Low Risk': '#22C55E', 'Medium Risk': '#EAB308', 'High Risk': '#EF4444'}
    order = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Layout Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Level Distribution")
        counts = df['Predicted Risk'].value_counts()
        labels = counts.index.tolist()
        sizes = counts.values.tolist()
        colors = [color_map[l] for l in labels]
        explode = [0.05 if l == 'High Risk' else 0 for l in labels]
        
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)
        
    with col2:
        st.subheader("Attendance vs Internal Score")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for risk in order:
            subset = df[df['Predicted Risk'] == risk]
            if len(subset) > 0:
                # Add jitter
                x_jitter = subset['attendance_pct'] + np.random.normal(0, 0.3, size=len(subset))
                y_jitter = subset['internal_score'] + np.random.normal(0, 0.3, size=len(subset))
                ax2.scatter(x_jitter, y_jitter, c=color_map[risk], label=risk, alpha=0.7)
        ax2.set_xlabel("Attendance %")
        ax2.set_ylabel("Internal Score")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        
    # Layout Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Internal Score Distribution")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        for risk in order:
            subset = df[df['Predicted Risk'] == risk]['internal_score']
            if len(subset) > 0:
                ax3.hist(subset, bins=20, alpha=0.6, color=color_map[risk], label=risk)
        ax3.set_xlabel("Internal Score")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
        
    with col4:
        st.subheader("Feature Importance")
        if model is not None and hasattr(model, 'feature_importances_'):
            features = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours', 'api']
            importances = model.feature_importances_
            
            # Sort ascending for horizontal bar chart so highest is at top
            indices = np.argsort(importances)
            
            sorted_features = [features[i] for i in indices]
            sorted_importances = importances[indices]
            
            # Determine colors: top 3 in descending order are the last 3 in ascending order
            bar_colors = ['#94A3B8'] * (len(features) - 3) + ['#14B8A6'] * 3
            
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            ax4.barh(sorted_features, sorted_importances, color=bar_colors)
            ax4.set_xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            st.info("Feature importance not available. Model not loaded correctly.")
            
    # Row 3 (Full width)
    st.subheader("Study Hours by Risk Category")
    fig5, ax5 = plt.subplots(figsize=(7, 4))
    
    # Prepare data for boxplot
    box_data = []
    valid_orders = []
    for risk in order:
        subset = df[df['Predicted Risk'] == risk]['study_hours']
        if len(subset) > 0:
            box_data.append(subset)
            valid_orders.append(risk)
            
    if box_data:
        bp = ax5.boxplot(box_data, patch_artist=True, tick_labels=valid_orders)
        for patch, risk in zip(bp['boxes'], valid_orders):
            patch.set_facecolor(color_map[risk])
            patch.set_alpha(0.7)
            
    ax5.set_ylabel("Study Hours")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)
    
    # High Risk Students List
    st.markdown("---")
    st.subheader("High Risk Students List")
    high_risk_df = df[df['Predicted Risk'] == 'High Risk']
    
    if len(high_risk_df) > 0:
        high_risk_sorted = high_risk_df.sort_values(by='attendance_pct', ascending=True)
        st.dataframe(high_risk_sorted, use_container_width=True)
    else:
        st.info("No high-risk students detected in this dataset.")
