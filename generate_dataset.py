import os
import random
import numpy as np
import pandas as pd

def generate_customer_churn_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic e-commerce & SaaS customer churn dataset."""
    np.random.seed(seed)
    random.seed(seed)
    
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 72, size=n_samples)
    contract_types = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_methods = np.random.choice(["Credit Card", "Bank Transfer", "Electronic Check", "PayPal"], size=n_samples, p=[0.35, 0.25, 0.25, 0.15])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    
    monthly_charges = []
    total_charges = []
    support_tickets = []
    satisfaction_scores = []
    churn_labels = []
    
    for i in range(n_samples):
        base_charge = random.uniform(20.0, 110.0)
        t = tenure_months[i]
        c_type = contract_types[i]
        
        # Tickets correlate with satisfaction
        tickets = np.random.poisson(lam=1.5 if c_type == "Month-to-Month" else 0.8)
        satisfaction = max(1, min(5, int(5 - (tickets * 0.7) + np.random.normal(0, 0.5))))
        
        # Churn probability model
        logit = -1.2
        if c_type == "Month-to-Month":
            logit += 1.3
        elif c_type == "Two-Year":
            logit -= 1.4
            
        logit += (base_charge - 65.0) * 0.02
        logit -= (t / 12.0) * 0.45
        logit += tickets * 0.4
        logit -= (satisfaction - 3.0) * 0.8
        
        prob_churn = 1.0 / (1.0 + np.exp(-logit))
        churn = 1 if random.random() < prob_churn else 0
        
        monthly_charges.append(round(base_charge, 2))
        total_charges.append(round(base_charge * t * random.uniform(0.95, 1.05), 2))
        support_tickets.append(tickets)
        satisfaction_scores.append(satisfaction)
        churn_labels.append(churn)
        
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure_months,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "paperless_billing": paperless_billing,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "satisfaction_score": satisfaction_scores,
        "churn": churn_labels
    })
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_customer_churn_data(5000)
    output_path = os.path.join("data", "customers.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path} with {len(df)} rows.")
    print(f"Overall Churn Rate: {df['churn'].mean() * 100:.2f}%")
