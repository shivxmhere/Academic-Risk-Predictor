import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Ensure the parent directory is in sys.path so we can import app.pages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pages.predict import show_predict_page
from app.pages.analytics import show_analytics_page
from app.pages.export import show_export_page

# 1. Page Config
st.set_page_config(page_title="Academic Risk Predictor", page_icon="🎓", layout="wide")

# 2. Load model and scaler
@st.cache_resource
def load_assets():
    try:
        # Assuming run from the academic-risk-predictor root dir
        model_path = os.path.join('models', 'random_forest_final.pkl')
        scaler_path = os.path.join('models', 'scaler.pkl')
        
        # Fallbacks if running directly from app dir
        if not os.path.exists(model_path):
            model_path = os.path.join('..', 'models', 'random_forest_final.pkl')
            scaler_path = os.path.join('..', 'models', 'scaler.pkl')
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model/scaler artifacts: {e}")
        return None, None

model, scaler = load_assets()
if model and scaler:
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler

# 4. Header with custom styling
st.markdown("""
<style>
.header-pill {
    background-color: #000080;
    color: white;
    padding: 10px 20px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    font-family: sans-serif;
}
.header-pill h1 {
    margin: 0;
    color: white;
    font-size: 2.5rem;
}
.header-pill p {
    margin: 0;
    font-size: 1.1rem;
    opacity: 0.9;
}
</style>
<div class="header-pill">
    <h1>🎓 Academic Risk Predictor</h1>
    <p>IIT Patna | Group 174 | CSDA 2025-26</p>
</div>
""", unsafe_allow_html=True)

# 3. Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📤 Upload Data", "🔮 Predict Risk", "📊 Analytics", "📄 Export Report"]
)

# 5. Upload Data Page
if page == "📤 Upload Data":
    st.subheader("Upload Student Data")
    st.markdown("Please upload a CSV file containing student academic records.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validation
            required_cols = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Validation Error! The following required columns are missing: {', '.join(missing_cols)}")
            else:
                st.session_state['raw_data'] = df
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Students", len(df))
                col2.metric("Columns Found", len(df.columns))
                col3.metric("Missing Values (NaNs)", df.isnull().sum().sum())
                
                st.write("### Data Preview (First 10 Rows)")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.success("Data loaded successfully. Go to 'Predict Risk' to run predictions.")
                
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

elif page == "🔮 Predict Risk":
    try:
        show_predict_page()
    except Exception as e:
        st.error(f"Predict Risk page is currently unavailable. Error: {e}")
        
elif page == "📊 Analytics":
    try:
        show_analytics_page()
    except Exception as e:
        st.error(f"Analytics page is currently unavailable. Error: {e}")
        
elif page == "📄 Export Report":
    try:
        show_export_page()
    except Exception as e:
        st.error(f"Export Report page is currently unavailable. Error: {e}")

else:
    # 6. Fallback welcome message
    st.write("Welcome to the Academic Risk Predictor. Please select an option from the sidebar to begin.")
