# Sales & Customer Insights Dashboard

**Author:** Amit Choudhury  
**Tools:** Python · pandas · NumPy · matplotlib · seaborn · scikit-learn  
**Domain:** Sales Analytics | Customer Segmentation | A/B Testing | ML

---

## 📌 Project Overview

An end-to-end analytics pipeline built on a 10,000-row sales dataset. Covers data cleaning, KPI analysis, customer segmentation, A/B testing, churn prediction modelling, and a 6-page dashboard output.

---

## 📊 What This Project Does

| Step | Description |
|------|-------------|
| **Data Generation** | Creates a realistic 10,000-row sales dataset with intentional nulls and dirty data |
| **Data Cleaning** | Reduces null values by 40%, fixes invalid entries, engineers new features |
| **KPI Analysis** | Calculates 8 KPIs: revenue, churn rate, AOV, acquisition rate, segmentation |
| **A/B Testing** | Compares discount vs standard pricing strategy across 1,200+ customers |
| **Churn Prediction** | Random Forest model with accuracy, precision, recall, F1-score evaluation |
| **Dashboard Output** | 6 PNG dashboard pages ready for presentation or LinkedIn post |

---

## 📁 Project Structure

```
project1_sales_dashboard/
│
├── generate_data.py        # Generates raw 10,000-row dataset
├── dashboard.py            # Main analytics + dashboard pipeline
├── raw_sales_data.csv      # Auto-generated on first run
│
├── page1_kpi_overview.png
├── page2_revenue_region_product.png
├── page3_trend_quarterly.png
├── page4_segmentation_churn.png
├── page5_ab_test.png
└── page6_model_evaluation.png
```

---

## 🚀 How to Run

```bash
# Step 1: Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Step 2: Generate dataset
python generate_data.py

# Step 3: Run full dashboard pipeline
python dashboard.py
```

---

## 📈 Key Findings

- **Underperforming products identified:** Cloud Sync & Report Engine (bottom 2 by revenue)
- **A/B Test result:** Discount-led strategy showed **+17% higher conversion rate** (statistically significant, p < 0.05)
- **Highest churn segment:** Startup (25.6%) — actionable for retention strategy
- **Churn model accuracy:** 84.6% using Random Forest Classifier

---

## 🛠️ Skills Demonstrated

- `pandas` — data cleaning, transformation, groupby aggregations
- `NumPy` — numerical operations, random data generation
- `scikit-learn` — RandomForestClassifier, train/test split, evaluation metrics
- `scipy.stats` — t-test for A/B testing significance
- `matplotlib` — multi-page dashboard visualizations
- Statistical analysis — KPI definition, segmentation, hypothesis testing
