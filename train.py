"""
Wine Quality Random Forest Classifier - Model Training Script
Trains a Random Forest model and outputs evaluation metrics.
"""

import os
import sys
from utils.data_utils import load_data, preprocess_data, get_dataset_info
from utils.model_utils import train_model, save_model

def main():
    print("=" * 60)
    print("  WINE QUALITY RANDOM FOREST CLASSIFIER - TRAINING SYSTEM ")
    print("=" * 60)
    
    # ── 1. Load Dataset ───────────────────────────────────────────────────────
    print("\n[Step 1/4] Loading dataset...")
    try:
        df = load_data()
        info = get_dataset_info(df)
        print(" -> Dataset loaded successfully!")
        print(f"   * Rows: {info['rows']}")
        print(f"   * Columns: {info['columns']}")
        print(f"   * Duplicates: {info['duplicates']}")
        print(f"   * Missing Values: {info['missing_values']}")
    except Exception as e:
        print(f"[-] Error loading dataset: {e}")
        sys.exit(1)
        
    # ── 2. Preprocess Data ────────────────────────────────────────────────────
    print("\n[Step 2/4] Preprocessing dataset...")
    try:
        X, y, df_processed = preprocess_data(df)
        good_count = sum(y == 1)
        bad_count = sum(y == 0)
        print(" -> Preprocessing completed!")
        print(f"   * Features count: {X.shape[1]}")
        print(f"   * Target classes: Good Wines (quality >= 7) = {good_count} ({good_count/len(y)*100:.2f}%)")
        print(f"   * Target classes: Bad/Avg Wines (quality < 7) = {bad_count} ({bad_count/len(y)*100:.2f}%)")
    except Exception as e:
        print(f"[-] Error preprocessing dataset: {e}")
        sys.exit(1)
        
    # ── 3. Train Model ────────────────────────────────────────────────────────
    print("\n[Step 3/4] Training Random Forest model...")
    try:
        # Default settings
        model, X_train, X_test, y_train, y_test, metrics = train_model(X, y)
        print(" -> Random Forest Model trained successfully!")
        print("-" * 40)
        print(f"   Accuracy Score:   {metrics['accuracy']}%")
        print(f"   Area Under ROC:   {metrics['roc_auc']}%")
        print(f"   Train Samples:    {metrics['train_samples']}")
        print(f"   Test Samples:     {metrics['test_samples']}")
        print("-" * 40)
        
        print("\nClassification Report:")
        print(metrics["classification_report_text"])
        
        print("\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print("               Predicted Bad  Predicted Good")
        print(f"True Bad:          {cm[0][0]:<14} {cm[0][1]}")
        print(f"True Good:         {cm[1][0]:<14} {cm[1][1]}")
        print("-" * 40)
    except Exception as e:
        print(f"[-] Error during training: {e}")
        sys.exit(1)
        
    # ── 4. Save Model ─────────────────────────────────────────────────────────
    print("\n[Step 4/4] Saving model binary...")
    try:
        save_path = save_model(model)
        print(f" -> Trained model successfully saved to: {os.path.abspath(save_path)}")
        print("=" * 60)
        print("  TRAINING SYSTEM RUN COMPLETE! READY FOR WEB DEPLOYMENT. ")
        print("=" * 60)
    except Exception as e:
        print(f"[-] Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
