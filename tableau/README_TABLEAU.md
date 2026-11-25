# Tableau Integration Guide

This directory documents how the datasets produced by this project are ingested into **Tableau Desktop / Tableau Public** to generate interactive executive BI dashboards.

---

## 📑 Data Source
- Path: `data/customers.csv`
- Format: Flat CSV (UTF-8)
- Granularity: One row per customer account

## 📊 Recommended Tableau Visualizations

### 1. Customer Churn by Contract Type & Payment Method
- **Dimensions**: `Contract Type`, `Payment Method`
- **Measures**: `CNT(Customer Id)`, `AVG(Churn)` (formatted as %)
- **Chart Type**: Stacked Bar Chart with dual-axis average churn percentage line.

### 2. Retention Hazard Cohort Matrix
- **Rows**: `Tenure (Bins of 6 Months)`
- **Columns**: `Paperless Billing`
- **Color**: `AVG(Churn)` with a Red-to-Green divergent palette.

### 3. Customer Satisfaction vs. Support Ticket Burden
- **X-axis**: `Support Tickets`
- **Y-axis**: `AVG(Satisfaction Score)`
- **Size**: `SUM(Monthly Charges)`
- **Tooltip**: Customer account details, lifetime value.

---

## 🔗 Publishing to Tableau Public
1. Open Tableau Desktop.
2. Select **Connect to File** -> **Text File** -> select `data/customers.csv`.
3. Create an **Extract** (`.hyper`) for performance.
4. Design dashboards using the metrics documented above.
5. Save & Publish to Tableau Public.
