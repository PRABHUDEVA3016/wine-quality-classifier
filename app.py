"""
Wine Quality Classification Streamlit Web Application
A professional college-level presentation project using Random Forest.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import datetime
import json

# Import local utility modules
from utils.data_utils import load_data, preprocess_data, get_dataset_info, COLUMN_DESCRIPTIONS
from utils.model_utils import train_model, predict_quality, model_exists, load_model
import utils.viz_utils as viz

# ──────────────────────────────────────────────────────────────────────────────
# Page Configurations & Styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality AI Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# Handcrafted Theme Presets
# ──────────────────────────────────────────────────────────────────────────────
THEMES = {
    "Royal Vineyard (Burgundy Dark)": {
        "bg_gradient": "linear-gradient(135deg, #1f0b0e 0%, #0d0405 100%)",
        "card_bg": "rgba(45, 15, 20, 0.45)",
        "card_border": "rgba(139, 0, 0, 0.35)",
        "accent": "#C0392B",
        "accent_gold": "#F1C40F",
        "text": "#F5EBEB",
        "subtext": "#d8b2b2",
        "shadow": "rgba(139, 0, 0, 0.18)",
        "plotly_template": "plotly_dark",
        "plotly_colors": ["#8B0000", "#C0392B", "#E74C3C", "#F39C12", "#F1C40F"],
        "text_color": "#F5EBEB",
        "bg_color": "rgba(0,0,0,0)",
        "heatmap_cmap": "RdYlGn"
    },
    "Emerald Cellar (Forest Green)": {
        "bg_gradient": "linear-gradient(135deg, #071c10 0%, #030d07 100%)",
        "card_bg": "rgba(15, 45, 25, 0.45)",
        "card_border": "rgba(46, 117, 72, 0.35)",
        "accent": "#2ecc71",
        "accent_gold": "#f1c40f",
        "text": "#EBF5EE",
        "subtext": "#a4c4b0",
        "shadow": "rgba(46, 117, 72, 0.18)",
        "plotly_template": "plotly_dark",
        "plotly_colors": ["#0F3820", "#1E6B3D", "#2ECC71", "#3498DB", "#F1C40F"],
        "text_color": "#EBF5EE",
        "bg_color": "rgba(0,0,0,0)",
        "heatmap_cmap": "YlGnBu"
    },
    "Sapphire Reserve (Classic Tech Blue)": {
        "bg_gradient": "linear-gradient(135deg, #08121e 0%, #03070d 100%)",
        "card_bg": "rgba(15, 25, 45, 0.45)",
        "card_border": "rgba(52, 152, 219, 0.35)",
        "accent": "#3498db",
        "accent_gold": "#f1c40f",
        "text": "#EBF2F5",
        "subtext": "#a4b9c4",
        "shadow": "rgba(52, 152, 219, 0.18)",
        "plotly_template": "plotly_dark",
        "plotly_colors": ["#0C2540", "#1C4E80", "#3498DB", "#9B59B6", "#F1C40F"],
        "text_color": "#EBF2F5",
        "bg_color": "rgba(0,0,0,0)",
        "heatmap_cmap": "mako"
    },
    "Golden Harvest (Champagne Light)": {
        "bg_gradient": "linear-gradient(135deg, #FAF7F0 0%, #EAE3D2 100%)",
        "card_bg": "rgba(255, 255, 255, 0.8)",
        "card_border": "rgba(180, 150, 100, 0.35)",
        "accent": "#865D36",
        "accent_gold": "#B8860B",
        "text": "#2C2C2C",
        "subtext": "#5C5C5C",
        "shadow": "rgba(180, 150, 100, 0.12)",
        "plotly_template": "plotly_white",
        "plotly_colors": ["#5C3D2E", "#865D36", "#AC7D4E", "#D4AF37", "#9E7676"],
        "text_color": "#2C2C2C",
        "bg_color": "rgba(0,0,0,0)",
        "heatmap_cmap": "coolwarm"
    }
}


def get_current_theme_key() -> str:
    """Evaluate session state theme select, supporting time-based automatic switching."""
    if "theme_select" not in st.session_state:
        st.session_state.theme_select = "Auto (Time-based)"
        
    selected = st.session_state.theme_select
    if selected == "Auto (Time-based)":
        hour = datetime.datetime.now().hour
        # Time-based bands:
        # 6 AM to 12 PM: Sapphire Reserve (Ocean Blue)
        # 12 PM to 5 PM: Golden Harvest (Champagne Light)
        # 5 PM to 9 PM: Emerald Cellar (Forest Green)
        # 9 PM to 6 AM: Royal Vineyard (Burgundy Dark)
        if 6 <= hour < 12:
            return "Sapphire Reserve (Classic Tech Blue)"
        elif 12 <= hour < 17:
            return "Golden Harvest (Champagne Light)"
        elif 17 <= hour < 21:
            return "Emerald Cellar (Forest Green)"
        else:
            return "Royal Vineyard (Burgundy Dark)"
    return selected


def apply_theme_css(config: dict):
    """Inject custom HTML and CSS to style the app's components dynamically."""
    bg = config["bg_gradient"]
    card_bg = config["card_bg"]
    card_border = config["card_border"]
    accent = config["accent"]
    accent_gold = config["accent_gold"]
    text = config["text"]
    subtext = config["subtext"]
    shadow = config["shadow"]
    is_light = (text == "#2C2C2C")
    
    css = f"""
    <style>
    /* Global Streamlit App background and text */
    .stApp {{
        background: {bg} !important;
        color: {text} !important;
    }}
    
    /* Global typography overrides */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown span {{
        color: {text} !important;
    }}
    
    /* Card containers */
    .wine-card {{
        background-color: {card_bg};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px {shadow};
        transition: all 0.3s ease;
    }}
    .wine-card:hover {{
        box-shadow: 0 8px 30px {accent}3d;
        transform: translateY(-2px);
    }}
    
    /* Glowing effect for predictions */
    .prediction-card-good {{
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, {card_bg} 100%);
        border: 2px solid {accent_gold};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.25);
        text-align: center;
    }}
    .prediction-card-bad {{
        background: linear-gradient(135deg, rgba(192, 57, 43, 0.15) 0%, {card_bg} 100%);
        border: 2px solid {accent};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 25px rgba(192, 57, 43, 0.25);
        text-align: center;
    }}
    
    /* Feature descriptions */
    .feature-desc {{
        font-size: 0.9em;
        color: {subtext};
        border-left: 3px solid {accent};
        padding-left: 10px;
        margin: 10px 0;
    }}
    
    /* Sidebar custom styling overrides */
    section[data-testid="stSidebar"] {{
        background-color: {"#F5EFEB" if is_light else "#07090D"} !important;
        border-right: 1px solid {card_border} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}
    
    /* Custom badges */
    .custom-badge {{
        background-color: {accent}22;
        color: {accent} !important;
        border: 1px solid {accent}88;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin: 4px;
    }}
    
    /* Buttons custom styling */
    .stButton>button {{
        background-color: {accent} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 10px {accent}3d !important;
        transition: all 0.3s ease !important;
    }}
    .stButton>button:hover {{
        background-color: {accent}ee !important;
        transform: scale(1.02) !important;
        box-shadow: 0 6px 15px {accent}66 !important;
    }}
    
    /* Input adjustments */
    div[data-baseweb="select"] {{
        background-color: {card_bg} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Initialization
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_cached_raw_data():
    """Retrieve raw dataset using local data utility helper."""
    try:
        return load_data()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return None

# Load dataset once
df_raw = get_cached_raw_data()

# Initialize session states
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "custom_model" not in st.session_state:
    st.session_state.custom_model = None
if "custom_metrics" not in st.session_state:
    st.session_state.custom_metrics = None

# Retrieve model (custom model trained in session state, otherwise disk loaded model)
def get_active_model_and_metrics():
    if st.session_state.custom_model is not None:
        return st.session_state.custom_model, st.session_state.custom_metrics
    
    # Otherwise try disk
    if model_exists():
        disk_model = load_model()
        # Mock up default metrics based on standard run
        if df_raw is not None:
            X, y, _ = preprocess_data(df_raw)
            # evaluate once and cache metrics in session state
            _, _, _, _, _, disk_metrics = train_model(X, y)
            st.session_state.custom_model = disk_model
            st.session_state.custom_metrics = disk_metrics
            return disk_model, disk_metrics
    return None, None

active_model, active_metrics = get_active_model_and_metrics()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h1 style='text-align: center; font-size: 1.8em; margin-bottom: 0px;'>🍷 Wine Quality AI</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9em; margin-top: 0px;'>Random Forest Classification Classifier</p>", unsafe_allow_html=True)
st.sidebar.divider()

# Aesthetic settings
with st.sidebar.expander("🎨 Dynamic Theme Engine", expanded=True):
    theme_options = [
        "Auto (Time-based)",
        "Royal Vineyard (Burgundy Dark)",
        "Emerald Cellar (Forest Green)",
        "Sapphire Reserve (Classic Tech Blue)",
        "Golden Harvest (Champagne Light)"
    ]
    st.selectbox(
        "Select Layout Accent Style",
        theme_options,
        key="theme_select"
    )
    theme_key = get_current_theme_key()
    theme_config = THEMES[theme_key]
    st.session_state.theme_config = theme_config
    
    # Inform user about auto theme
    if st.session_state.theme_select == "Auto (Time-based)":
        st.caption(f"🕒 Automatically applied theme based on system time: **{theme_key.split(' (')[0]}**")
    else:
        st.caption(f"🎨 Manually locked theme: **{theme_key.split(' (')[0]}**")

# Side Navigation Menu
st.sidebar.divider()
pages = {
    "🏠 Project Introduction": "home",
    "📅 Dataset Explorer": "dataset",
    "🤖 Model Training Control": "training",
    "🔮 Predict Wine Quality": "prediction",
    "📊 Advanced Visualizations": "viz"
}
selected_page = st.sidebar.radio("Navigation Menu", list(pages.keys()))
page_id = pages[selected_page]

# Footer Info
st.sidebar.divider()
st.sidebar.caption("👨‍💻 **College Project Submission**")
st.sidebar.caption("AI/ML Viva Presentation - 2026")

# Apply Dynamic CSS Styles
apply_theme_css(theme_config)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME PAGE
# ──────────────────────────────────────────────────────────────────────────────
if page_id == "home":
    st.title("🍷 Wine Quality Classification Using Random Forest")
    st.subheader("An Elegant Machine Learning System for Quality Grading")
    
    # Big Hero Container
    st.markdown(
        f"""
        <div class='wine-card'>
            <h3 style='margin-top:0;'>📊 Project Overview</h3>
            <p>Welcome to the <b>Wine Quality Classifier</b> dashboard. This professional-grade web application predicts whether a red wine is of 
            <b>High Quality (Good Wine, scored ≥ 7)</b> or <b>Low/Medium Quality (Bad Wine, scored < 7)</b> based on its objective physicochemical properties. 
            Using a robust ensemble model—the <b>Random Forest Classifier</b>—our app provides high accuracy, predictive breakdown, and instant 
            visual diagnostics.</p>
            <p><b>Target Split Standard:</b> High Quality = Quality Score &ge; 7 | Bad/Average Quality = Quality Score &le; 6.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class='wine-card' style='height: 480px;'>
                <h3>🌲 How Random Forest Works (Ensemble Bagging)</h3>
                <p>The <b>Random Forest</b> is a powerful machine learning ensemble algorithm that combines multiple decision trees to create a highly accurate classifier.</p>
                <ul>
                    <li><b>Bootstrap Aggregation (Bagging):</b> It randomly samples the dataset with replacement to train distinct decision trees, making the overall model highly robust against noise.</li>
                    <li><b>Feature Randomness:</b> While constructing each split in a tree, it selects a random subset of chemical attributes. This prevents a single dominant feature (like alcohol) from dominating every single decision branch, adding critical diversity.</li>
                    <li><b>Majority Voting:</b> For a new sample, each individual tree votes. The category with the majority votes becomes the final output prediction. This reduces overfitting dramatically.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class='wine-card' style='height: 480px;'>
                <h3>🎓 Presentation & Viva Checklist</h3>
                <p>If you are presenting this project in front of college evaluators, make sure to highlight the following milestones:</p>
                <ol>
                    <li><b>Feature Engineering:</b> Describe the chemical components (e.g. <i>volatile acidity</i> representing vinegar taste, <i>alcohol content</i> reflecting body/sugar content).</li>
                    <li><b>Balanced Stratification:</b> Our training script automatically handles stratification during the train/test split to maintain class distributions.</li>
                    <li><b>Confusion Matrix:</b> Explain the True Positives (Good wines predicted as good) and True Negatives (Bad wines predicted as bad) quadrants.</li>
                    <li><b>Interactive Training:</b> Leverage the "Model Training" tab to demonstrate live training with custom trees. Show how increasing n_estimators affects performance metrics.</li>
                    <li><b>Theme Engine:</b> Point out the custom CSS styling and time-based automatic shifting showcasing backend capability!</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: DATASET EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
elif page_id == "dataset":
    st.title("📅 Dataset Exploration & Analysis")
    st.subheader("Physicochemical Profile of Red Wine Samples")
    
    if df_raw is None:
        st.error("No dataset loaded.")
    else:
        info = get_dataset_info(df_raw)
        
        # Key Metrics Columns
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Dataset Rows", info["rows"])
        mcol2.metric("Dataset Features", info["columns"] - 1)  # excluding target
        mcol3.metric("Missing Values", info["missing_values"])
        mcol4.metric("Duplicate Samples", info["duplicates"])
        
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📝 Data Description & Units", "🔍 Raw Data Preview", "📊 Quality Distribution"])
        
        with tab1:
            st.markdown("### Chemical Properties Reference Card")
            st.markdown("Below are the definitions, chemical relevance, and typical measurement units for all features in the dataset:")
            
            for key, desc in COLUMN_DESCRIPTIONS.items():
                st.markdown(
                    f"""
                    <div class='feature-desc'>
                        <b>{key.title()}</b>: {desc}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
        with tab2:
            st.markdown("### Interactive Dataset Preview")
            row_slider = st.slider("Select number of rows to preview:", 5, 200, 15)
            st.dataframe(df_raw.head(row_slider), use_container_width=True)
            
            st.markdown("### Dataset Summary Statistics")
            st.dataframe(df_raw.describe(), use_container_width=True)
            
        with tab3:
            st.markdown("### Target Score Distribution")
            st.markdown(
                "Wines are scored by expert tasters between 0 and 10. Our model classifies scores **&ge; 7** as High Quality (Good) "
                "and **&le; 6** as Low/Medium Quality (Bad)."
            )
            dist_fig = viz.plot_quality_distribution(df_raw)
            st.plotly_chart(dist_fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
elif page_id == "training":
    st.title("🤖 Random Forest Model Training")
    st.subheader("Tune Hyperparameters and Train the Model in Real-Time")
    
    if df_raw is None:
        st.error("No dataset available to train.")
    else:
        st.markdown(
            """
            <div class='wine-card'>
                <h4>🔧 Hyperparameter Tuning Panel</h4>
                Adjust the sliding values below to configure the Random Forest Classifier. In a viva presentation, you can demonstrate 
                how changing these hyperparameters affects the classification performance.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Two Column Configuration Inputs
        ccol1, ccol2 = st.columns(2)
        with ccol1:
            n_estimators = st.slider("🌲 Number of Trees (n_estimators)", 10, 500, 200, step=10, 
                                     help="The number of decision trees in the forest.")
            max_depth = st.slider("📐 Maximum Tree Depth (max_depth)", 2, 30, 12, 
                                  help="The maximum depth of the individual decision trees.")
            test_size_pct = st.slider("🧪 Test Split Ratio (%)", 10, 50, 20, 
                                      help="The proportion of the dataset to hold out for testing.")
            
        with ccol2:
            min_samples_split = st.slider("✂ Minimum Samples to Split", 2, 10, 5, 
                                          help="The minimum number of samples required to split an internal node.")
            min_samples_leaf = st.slider("🍃 Minimum Samples in Leaf Node", 1, 10, 2, 
                                         help="The minimum number of samples required to be at a leaf node.")
            random_state = st.number_input("🎲 Random State (Seed)", 0, 1000, 42, 
                                           help="Ensures reproducibility of the train/test split and tree bootstrapping.")
            
        st.divider()
        
        if st.button("🚀 Train Random Forest Classifier"):
            with st.spinner("Preparing dataset... growing decision trees... checking votes..."):
                X, y, _ = preprocess_data(df_raw)
                
                # Call modular train model with custom config
                custom_model, X_tr, X_te, y_tr, y_te, custom_metrics = train_model(
                    X, y,
                    test_size=(test_size_pct / 100.0),
                    random_state=random_state,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf
                )
                
                # Save to session state to make prediction page dynamic!
                st.session_state.custom_model = custom_model
                st.session_state.custom_metrics = custom_metrics
                
                # Force updates
                active_model = custom_model
                active_metrics = custom_metrics
                
                st.toast("🌲 Model Trained Successfully!", icon="🎉")
                st.success("🎉 Random Forest classifier successfully trained in real-time!")
                
        # If model trained, display metrics
        if active_metrics is not None:
            st.markdown("### 📈 Live Classification Performance Diagnostics")
            
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("Testing Accuracy Score", f"{active_metrics['accuracy']}%")
            pcol2.metric("Area Under ROC curve", f"{active_metrics['roc_auc']}%")
            pcol3.metric("Training Samples", active_metrics["train_samples"])
            pcol4.metric("Holdout Test Samples", active_metrics["test_samples"])
            
            diag_col1, diag_col2 = st.columns([2, 3])
            
            with diag_col1:
                st.markdown("#### Annotated Confusion Matrix")
                cm_fig = viz.plot_confusion_matrix(active_metrics["confusion_matrix"])
                st.plotly_chart(cm_fig, use_container_width=True)
                
            with diag_col2:
                st.markdown("#### Multi-Class Precision & Recall Breakdown")
                
                # Format classification report into beautiful table
                rep = active_metrics["classification_report"]
                rep_df = pd.DataFrame(rep).transpose().iloc[:3, :3] # Keep class 0, 1, and accuracy rows
                rep_df = rep_df.rename(index={"0": "Bad/Avg Wine (0)", "1": "Good Wine (1)"})
                
                st.dataframe(rep_df.style.background_gradient(cmap="Reds"), use_container_width=True)
                
                # Explanation
                st.markdown(
                    """
                    > **💡 Understanding the Metrics:**
                    > - **Precision:** Out of all wines predicted as Good, how many were actually Good? (High precision means low false alarms).
                    > - **Recall:** Out of all actual Good wines, how many did the Random Forest successfully catch? (High recall means low missed opportunities).
                    > - **F1-Score:** The harmonic mean of precision and recall. A great single measure of classifier performance.
                    """
                )
        else:
            st.info("💡 No custom model trained yet. Click the button above to train your Random Forest model dynamically, or prediction will use the pre-trained default model.")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4: PREDICTION DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
elif page_id == "prediction":
    st.title("🔮 Interactive Quality Predictor")
    st.subheader("Enter Physicochemical Metrics to Determine Wine Quality")
    
    if active_model is None:
        st.warning("⚠️ No machine learning model is loaded! Please go to the 'Model Training Control' page and train the model first.")
    else:
        st.markdown(
            """
            <div class='wine-card'>
                <h4>🍷 Set Chemical Input Parameters</h4>
                Sliders are bounded within realistic minimum and maximum ranges from the dataset. 
                Configure the sliders below and click the prediction button to get an instant prediction.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Sliders in 3 columns
        sc1, sc2, sc3 = st.columns(3)
        
        with sc1:
            fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, 8.3, 0.1,
                                      help="Tartaric acid, prevents spoilage and affects taste structure.")
            volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.1, 1.6, 0.52, 0.01,
                                         help="Acetic acid, high levels cause vinegar-like off flavours.")
            citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.0, 0.27, 0.01,
                                    help="Adds freshness and crispness, lifts flat flavours.")
            residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.5, 16.0, 2.2, 0.1,
                                       help="Natural sugars left after fermentation completes.")
            
        with sc2:
            chlorides = st.slider("Chlorides (g/dm³)", 0.01, 0.62, 0.079, 0.001,
                                  help="Sodium chloride, represents mineral salts content.")
            free_so2 = st.slider("Free Sulfur Dioxide (mg/dm³)", 1.0, 75.0, 14.0, 1.0,
                                 help="Free SO₂ prevents bacterial spoilage and oxidation.")
            total_so2 = st.slider("Total Sulfur Dioxide (mg/dm³)", 6.0, 290.0, 46.0, 1.0,
                                  help="Total SO₂; high amounts cause sensory defects.")
            density = st.slider("Relative Density (g/cm³)", 0.990, 1.004, 0.9967, 0.0001,
                                help="Density of wine relative to water, correlates with alcohol.")
            
        with sc3:
            ph = st.slider("pH Level", 2.7, 4.1, 3.31, 0.01,
                           help="Wine acidity scale (0 represents acidic, 14 alkaline).")
            sulphates = st.slider("Sulphates Additive (g/dm³)", 0.3, 2.0, 0.62, 0.01,
                                  help="Wine additive that releases free SO₂ gas.")
            alcohol = st.slider("Alcohol Content (% vol)", 8.0, 15.0, 10.2, 0.1,
                                help="Alcohol by volume percentage, contributes to body.")
            
        st.divider()
        
        # Perform Prediction
        inputs = np.array([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_so2, total_so2, density, ph, sulphates, alcohol
        ])
        
        if st.button("🍷 Predict Wine Quality"):
            pred, conf, probas = predict_quality(active_model, inputs)
            
            # Predict labels
            label = "Good Wine" if pred == 1 else "Bad Wine"
            glow_class = "prediction-card-good" if pred == 1 else "prediction-card-bad"
            badge_icon = "🥇" if pred == 1 else "⚠️"
            
            st.markdown("### Prediction Diagnostic Report")
            
            res_col1, res_col2 = st.columns([3, 2])
            
            with res_col1:
                st.markdown(
                    f"""
                    <div class='{glow_class}'>
                        <h2>{badge_icon} Classification: {label.upper()}</h2>
                        <h4 style='opacity: 0.9; margin: 5px 0;'>Model Confidence Score: {conf}%</h4>
                        <p style='font-size:0.95em;'>Wines classified as Good Wine are statistically determined by 
                        our Random Forest forest to have a high sensory quality score (&ge; 7).</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Single Prediction Download Report Extra Feature
                report_data = {
                    "Analysis Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Inputs": {
                        "fixed_acidity": fixed_acidity, "volatile_acidity": volatile_acidity, "citric_acid": citric_acid,
                        "residual_sugar": residual_sugar, "chlorides": chlorides, "free_sulfur_dioxide": free_so2,
                        "total_sulfur_dioxide": total_so2, "density": density, "ph": ph, "sulphates": sulphates, "alcohol": alcohol
                    },
                    "PredictionResult": {
                        "class_label": label,
                        "class_code": pred,
                        "confidence_pct": conf,
                        "probabilities": probas
                    }
                }
                
                report_str = json.dumps(report_data, indent=4)
                st.download_button(
                    "📥 Download Prediction Report (JSON)",
                    report_str,
                    file_name="wine_quality_report.json",
                    mime="application/json"
                )
                
            with res_col2:
                # Gauge indicator
                gauge_fig = viz.plot_confidence_gauge(conf, label)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
            # Log to Prediction History Table
            history_record = {
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Alcohol": alcohol,
                "Volatile Acidity": volatile_acidity,
                "Sulphates": sulphates,
                "pH": ph,
                "Free SO2": free_so2,
                "Predicted Quality": "Good (>=7)" if pred == 1 else "Bad/Avg (<7)",
                "Confidence": f"{conf}%"
            }
            st.session_state.prediction_history.insert(0, history_record)
            
        # Display Prediction History
        if len(st.session_state.prediction_history) > 0:
            st.divider()
            st.markdown("### 📊 Prediction History Logs")
            st.markdown("This table maintains a session log of the predicted samples for quick comparative review.")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Export prediction history to CSV
            csv_str = history_df.to_csv(index=False)
            st.download_button(
                "📥 Export Full Prediction History to CSV",
                csv_str,
                file_name="predicted_history_logs.csv",
                mime="text/csv"
            )

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 5: ADVANCED VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
elif page_id == "viz":
    st.title("📊 Model & Data Advanced Visualizations")
    st.subheader("Statistical Diagnostics and Feature Importance Suite")
    
    if df_raw is None:
        st.error("No dataset available to visualize.")
    else:
        vtab1, vtab2, vtab3, vtab4 = st.tabs([
            "🌲 Feature Importance", 
            "📈 Performance Analytics", 
            "🌡️ Feature Correlations", 
            "📦 Distribution Diagnostics"
        ])
        
        with vtab1:
            st.markdown("### Forest Feature Importances")
            st.markdown(
                "Feature importance measures how much each chemical property contributes to splitting node decisions in the ensemble. "
                "Higher values represent stronger predictive features."
            )
            if active_metrics is not None:
                fi_fig = viz.plot_feature_importance(active_metrics["feature_importance"])
                st.plotly_chart(fi_fig, use_container_width=True)
                st.markdown(
                    """
                    > **💡 Key Insights for Viva presentation:**
                    > - **Alcohol Content** typically scores as the highest predictor. Higher alcohol concentration correlates strongly with higher quality wines.
                    > - **Volatile Acidity** (acetic acid) is also a strong predictor, but has a negative relationship (higher volatile acidity implies lower quality).
                    > - **Sulphates** add SO₂ which acts as a preservative, preventing off-flavours, hence contributing positively to quality.
                    """
                )
            else:
                st.info("Train the model on the 'Model Training Control' page first to see dynamic feature importances.")
                
        with vtab2:
            st.markdown("### Comparative Performance Scores")
            if active_metrics is not None:
                perf_fig = viz.plot_metrics_bar(active_metrics)
                st.plotly_chart(perf_fig, use_container_width=True)
            else:
                st.info("Train the model on the 'Model Training Control' page first to see dynamic model metrics.")
                
        with vtab3:
            st.markdown("### Linear Correlation Heatmap")
            st.markdown(
                "The correlation heatmap measures linear dependencies. A score near +1 represents a strong positive correlation, "
                "-1 represents a strong negative correlation, and 0 means no linear correlation."
            )
            with st.spinner("Generating correlation calculations..."):
                corr_fig = viz.plot_correlation_heatmap(df_raw)
                st.pyplot(corr_fig)
                
        with vtab4:
            st.markdown("### Custom Distribution Boxplots & KDE Histograms")
            st.markdown("Compare the distributions of critical chemical features between **Good Wine** and **Bad Wine**.")
            
            # Select feature boxplots
            feat_names = [col for col in df_raw.columns if col not in ("quality", "quality_label")]
            
            box_fig = viz.plot_boxplots(df_raw, feat_names)
            st.plotly_chart(box_fig, use_container_width=True)
            
            st.divider()
            
            st.markdown("#### Histograms and Kernel Density Estimates (KDE)")
            with st.spinner("Generating feature density distributions..."):
                dist_grid_fig = viz.plot_feature_distributions(df_raw, feat_names)
                st.pyplot(dist_grid_fig)
