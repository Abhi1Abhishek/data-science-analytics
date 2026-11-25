import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
import joblib

def build_and_evaluate_models(csv_path: str = "data/customers.csv"):
    """Train, evaluate, and benchmark Scikit-learn predictive models."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run generate_dataset.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Feature Engineering
    df["charges_per_tenure"] = df["total_charges"] / (df["tenure_months"] + 1)
    df["ticket_frequency"] = df["support_tickets"] / (df["tenure_months"] + 1)
    
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]
    
    num_features = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "satisfaction_score", "charges_per_tenure", "ticket_frequency"]
    cat_features = ["contract_type", "payment_method", "paperless_billing"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_auc = 0.0
    best_model_name = None
    best_fitted_pipeline = None
    
    print("\n" + "=" * 60)
    print("🤖 TRAINING & BENCHMARKING SCIKIT-LEARN MODELS")
    print("=" * 60)
    
    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        # Subsample ROC points for clean JSON export
        step = max(1, len(fpr) // 50)
        roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::step], tpr[::step])]
        
        results[name] = {
            "cv_roc_auc_mean": round(float(np.mean(cv_scores)), 4),
            "cv_roc_auc_std": round(float(np.std(cv_scores)), 4),
            "test_accuracy": round(float(acc), 4),
            "test_precision": round(float(prec), 4),
            "test_recall": round(float(rec), 4),
            "test_f1": round(float(f1), 4),
            "test_roc_auc": round(float(auc), 4),
            "roc_curve": roc_points,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"[{name}]")
        print(f"  CV ROC-AUC: {results[name]['cv_roc_auc_mean']:.4f} (+/- {results[name]['cv_roc_auc_std']:.4f})")
        print(f"  Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_fitted_pipeline = pipe
            
    # Extract Feature Importances for the Tree-based model
    encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_names
    
    feature_importances = []
    if hasattr(best_fitted_pipeline.named_steps["classifier"], "feature_importances_"):
        raw_importances = best_fitted_pipeline.named_steps["classifier"].feature_importances_
        for feat, imp in sorted(zip(all_feature_names, raw_importances), key=lambda x: x[1], reverse=True)[:10]:
            feature_importances.append({"feature": feat, "importance": round(float(imp), 4)})
            
    # Save best model
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "best_churn_model.joblib")
    joblib.dump(best_fitted_pipeline, model_save_path)
    print(f"Saved best model ({best_model_name}) to {model_save_path}")
    
    # Export metrics for D3.js dashboard
    os.makedirs("dashboard", exist_ok=True)
    summary_export = {
        "best_model": best_model_name,
        "models": results,
        "feature_importances": feature_importances,
        "sample_points": df[["tenure_months", "monthly_charges", "churn", "satisfaction_score"]].sample(120, random_state=42).to_dict(orient="records")
    }
    
    metrics_path = os.path.join("dashboard", "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_export, f, indent=2)
    print(f"Exported visualization metrics to {metrics_path}")
    
    return summary_export

if __name__ == "__main__":
    build_and_evaluate_models()
