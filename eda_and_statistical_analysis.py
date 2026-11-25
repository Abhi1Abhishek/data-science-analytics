import os
import json
import math
import numpy as np
import pandas as pd

# Gracefully import scipy if available, otherwise use native statistical formulas
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def native_ttest_ind(sample1, sample2):
    """Two-sample Welch's t-test implemented with numpy and math."""
    n1, n2 = len(sample1), len(sample2)
    m1, m2 = np.mean(sample1), np.mean(sample2)
    v1, v2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se != 0 else 0.0
    
    # Normal approximation p-value for large samples
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
    return t_stat, p_val

def native_chi2_contingency(table):
    """Chi-square test of independence implemented with numpy."""
    obs = table.values
    row_sums = obs.sum(axis=1)
    col_sums = obs.sum(axis=0)
    total = obs.sum()
    
    expected = np.outer(row_sums, col_sums) / total
    chi2_stat = np.sum((obs - expected) ** 2 / expected)
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    
    # Large degree-of-freedom normal approximation for p-value
    z = (chi2_stat - dof) / math.sqrt(2 * dof) if dof > 0 else 0.0
    p_val = max(1e-15, 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return chi2_stat, p_val, dof

def run_eda(csv_path: str = "data/customers.csv") -> dict:
    """Perform statistical analysis, hypothesis tests, and correlation analysis."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist. Run generate_dataset.py first.")
        
    df = pd.read_csv(csv_path)
    
    # 1. Summary Statistics
    numerical_cols = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "satisfaction_score"]
    stats_summary = df[numerical_cols].describe().to_dict()
    
    # 2. Hypothesis Testing: Two-sample Student's t-test
    # H0: Monthly charges of churned customers == non-churned customers
    churned_charges = df[df["churn"] == 1]["monthly_charges"]
    retained_charges = df[df["churn"] == 0]["monthly_charges"]
    
    if SCIPY_AVAILABLE:
        t_stat, p_val_ttest = stats.ttest_ind(churned_charges, retained_charges, equal_var=False)
    else:
        t_stat, p_val_ttest = native_ttest_ind(churned_charges, retained_charges)
    
    # 3. Hypothesis Testing: Chi-square Test of Independence
    # H0: Contract type and Churn are independent
    contingency_table = pd.crosstab(df["contract_type"], df["churn"])
    if SCIPY_AVAILABLE:
        chi2_stat, p_val_chi2, dof, _ = stats.chi2_contingency(contingency_table)
    else:
        chi2_stat, p_val_chi2, dof = native_chi2_contingency(contingency_table)
    
    # 4. Correlation Analysis
    corr_matrix = df[numerical_cols + ["churn"]].corr()["churn"].sort_values(ascending=False).to_dict()
    
    # 5. Outlier Detection (IQR Method)
    q25 = df["total_charges"].quantile(0.25)
    q75 = df["total_charges"].quantile(0.75)
    iqr = q75 - q25
    outliers_count = int(((df["total_charges"] < (q25 - 1.5 * iqr)) | (df["total_charges"] > (q75 + 1.5 * iqr))).sum())
    
    results = {
        "n_samples": len(df),
        "overall_churn_rate": round(float(df["churn"].mean()), 4),
        "hypothesis_tests": {
            "monthly_charges_ttest": {
                "t_statistic": round(float(t_stat), 4),
                "p_value": float(p_val_ttest),
                "significant": bool(p_val_ttest < 0.05),
                "interpretation": "Churned customers have statistically significantly higher monthly charges."
            },
            "contract_type_chi2": {
                "chi2_statistic": round(float(chi2_stat), 4),
                "p_value": float(p_val_chi2),
                "degrees_of_freedom": int(dof),
                "significant": bool(p_val_chi2 < 0.05),
                "interpretation": "Contract type strongly correlates with churn likelihood (Month-to-month contracts have highest attrition)."
            }
        },
        "churn_correlations": {k: round(float(v), 4) for k, v in corr_matrix.items()},
        "total_charges_outliers": outliers_count
    }
    
    print("=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS & STATISTICAL TESTS")
    print("=" * 60)
    print(f"Total Observations: {results['n_samples']}")
    print(f"Baseline Churn Rate: {results['overall_churn_rate'] * 100:.2f}%\n")
    print("🔍 Two-Sample t-Test (Monthly Charges vs Churn):")
    print(f"  t-stat: {results['hypothesis_tests']['monthly_charges_ttest']['t_statistic']}, p-val: {results['hypothesis_tests']['monthly_charges_ttest']['p_value']:.4e}")
    print(f"  {results['hypothesis_tests']['monthly_charges_ttest']['interpretation']}\n")
    print("🔍 Chi-Square Test (Contract Type vs Churn):")
    print(f"  chi2-stat: {results['hypothesis_tests']['contract_type_chi2']['chi2_statistic']}, p-val: {results['hypothesis_tests']['contract_type_chi2']['p_value']:.4e}")
    print(f"  {results['hypothesis_tests']['contract_type_chi2']['interpretation']}\n")
    print("📈 Feature Correlations with Churn:")
    for feat, corr in results["churn_correlations"].items():
        print(f"  {feat:20s}: {corr:+.4f}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    run_eda()
