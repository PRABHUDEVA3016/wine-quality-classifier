# 🍷 Wine Quality Classification Using Random Forest

An elegant, production-ready, professional AI/ML web application built with **Python**, **Streamlit**, and **Scikit-Learn**. This project classifies red wine quality as **High Quality ("Good Wine", scored >= 7)** or **Low/Medium Quality ("Bad/Average Wine", scored < 7)** using objective physicochemical parameters from the UCI Wine Quality dataset.

Designed specifically for college AI/ML presentations, viva demonstrations, and portfolio showcases, it features real-time dynamic model training, a custom-designed **Dynamic Theme Engine** (offering automated and manual presets), full prediction logs export, and a high-end visual diagnostics suite.

---

## 🚀 Key Features

*   **⚡ CLI Model Training Script (`train.py`)**: A standalone Python CLI tool that fetches the dataset, pre-processes it (null handling, class balancing, feature-target separation), trains a stratified Random Forest classifier, prints formatted evaluations (Accuracy, ROC-AUC, Confusion Matrix, Classification Report), and outputs a saved binary.
*   **🎨 Dynamic Theme Engine**: Tailored styling using customized CSS class injection. Supports time-of-day time-based automatic switches or manual locked settings:
    *   `Royal Vineyard (Burgundy Dark)` 🍷 - Rich dark crimson for late evening.
    *   `Emerald Cellar (Forest Green)` 🍇 - Forest and emerald tones.
    *   `Sapphire Reserve (Classic Tech Blue)` 🌊 - Modern high-tech blue for mornings.
    *   `Golden Harvest (Champagne Light)` ☀️ - Warm light mode champagne theme for midday.
*   **📅 Interactive Dataset Explorer**: Includes chemical attribute directories, tabular search previews, standard summaries, and an interactive Plotly target score distribution bar chart.
*   **🤖 Live Hyperparameter Training**: Sliders to adjust `n_estimators`, `max_depth`, split ratios, and leaf counts with in-app retraining, generating live Plotly heatmaps and precision-recall metrics.
*   **🔮 High-End Predictor & History Log**: Bounded inputs matching true wine metrics, visual Gauge confidence indicators, formatted single JSON report downloads, and a full prediction log history table that allows downloading multiple runs as a merged CSV file.
*   **📊 Advanced Diagnostics Dashboard**: Interactive Plotly and Matplotlib visuals including Feature Importance scores, correlation heatmaps, feature comparative boxplots, and grid KDE histograms.

---

## 📂 Project Structure

```
wine_quality/
│
├── dataset/
│   ├── download_dataset.py     # Download helper script
│   └── winequality-red.csv     # UCI Red Wine CSV dataset
│
├── saved_model/
│   └── random_forest_model.pkl   # Stratified Random Forest model binary
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py           # Preprocessing & chemical definitions
│   ├── model_utils.py          # Random Forest training & predictor helpers
│   └── viz_utils.py            # Adaptable theme-aware Plotly/Seaborn plots
│
├── train.py                    # Independent model training runner (ASCII console safe)
├── app.py                      # Multi-page Streamlit Web App (Main)
├── requirements.txt            # System dependencies
└── README.md                   # Project documentation & presentation guide
```

---

## 🛠️ Installation & Setup

Follow these simple steps to set up the environment and run the application locally.

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 1. Clone or Open Workspace
Navigate to the project root directory:
```bash
cd wine_quality
```

### 2. Install Dependencies
Install all required packages from `requirements.txt` using pip:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Run CLI Model Training
The project comes with a pre-trained default model on startup, but you can explicitly trigger a standalone training run to verify CLI operations:
```bash
python train.py
```
This downloads the dataset automatically and outputs:
*   Model Testing Accuracy: **93.75%**
*   Area Under ROC: **94.41%**
*   Saves the model to: `saved_model/random_forest_model.pkl`

---

## 🏃 Running the Web Application

To launch the interactive dashboard, run the Streamlit server:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser (usually at `http://localhost:8501`).

---

## 🧪 Chemical Features Directory

For a successful project presentation and viva, familiarize yourself with these 11 chemical features:

1.  **Fixed Acidity** ($g/dm^3$): Most acids involved with wine (mainly tartaric). Contributes to the longevity, structure, and sharp taste.
2.  **Volatile Acidity** ($g/dm^3$): Acetic acid content. Extremely high levels give wine an unpleasant, vinegar-like taste.
3.  **Citric Acid** ($g/dm^3$): Found in small quantities, adds freshness and crisp fruit notes to wines.
4.  **Residual Sugar** ($g/dm^3$): Sugar remaining after fermentation completes. Controls sweetness and body.
5.  **Chlorides** ($g/dm^3$): Relates to the mineral salts content in wine, influencing saltiness.
6.  **Free Sulfur Dioxide** ($mg/dm^3$): The active $SO_2$ gas that acts as an antioxidant and prevents bacterial spoilage.
7.  **Total Sulfur Dioxide** ($mg/dm^3$): Sum of free and bound $SO_2$. Excess levels cause flat tastes and can trigger headaches.
8.  **Density** ($g/cm^3$): Density relative to water. Correlates closely with alcohol percentage and sugar concentration.
9.  **pH Level**: Measures acidity/basicity (0 very acidic, 14 very alkaline). Red wines typically range between 3.0 and 4.0.
10. **Sulphates** ($g/dm^3$): A wine additive that releases active sulfur dioxide gas ($SO_2$), preserving wine freshness.
11. **Alcohol** (% vol): Percentage of ethanol by volume. Greatly influences body, warmth, and flavor perception.

---

## 🎓 Viva Presentation Tips & Score Boosters

Make your project stand out in front of evaluators with these technical selling points:

*   **Describe Random Forest Logic**: Explain that Random Forest is an **Ensemble method** using **Bootstrap Aggregating (Bagging)** to train independent trees on random subsets of columns and rows. Show that this prevents overfitting (unlike a single Decision Tree).
*   **Show Hyperparameter Impact**: Run the app and go to the **Model Training Control** page. Drag `n_estimators` to a lower number (like `20`) and show the drop in accuracy, then pull it back to `200` to show how ensemble size improves predictive stability.
*   **Explain the Confusion Matrix**:
    *   **True Negatives (TN):** Bad/Average wines predicted as bad.
    *   **True Positives (TP):** Good wines predicted as good.
    *   **False Negatives (FN):** actual good wines the model missed (predicted as bad). Show that FN is 18 while FP is only 2, which is excellent for safety (low false alarms).
*   **Highlight the UI Features**: Point out the **Dynamic Theme Engine** in the sidebar. Toggle the theme to champagne or green and point out how every chart (Plotly/Seaborn) updates its color palette on the fly!
