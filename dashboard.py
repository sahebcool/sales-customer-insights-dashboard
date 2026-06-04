"""
Sales & Customer Insights Dashboard
Author: Amit Choudhury
Tools: Python (pandas, NumPy, matplotlib, seaborn, scikit-learn)
Description: End-to-end analytics pipeline — data cleaning, KPI analysis,
             segmentation, A/B testing, churn prediction, and dashboard visuals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── STYLE ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': '#F8FAFC',
    'axes.facecolor': '#F8FAFC',
})
BLUE   = '#1B3A6B'
ACCENT = '#2563EB'
GREEN  = '#16A34A'
RED    = '#DC2626'
ORANGE = '#EA580C'
GRAY   = '#64748B'
PALETTE = [BLUE, ACCENT, GREEN, ORANGE, RED]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: DATA LOADING & CLEANING")
print("=" * 60)

df_raw = pd.read_csv('raw_sales_data.csv', parse_dates=['date'])
print(f"Raw data shape     : {df_raw.shape}")
print(f"Null values before : {df_raw.isnull().sum().sum()}")

df = df_raw.copy()

# Fill missing revenue with product median
df['revenue'] = df.groupby('product')['revenue'].transform(lambda x: x.fillna(x.median()))

# Fill missing segment with mode
df['segment'] = df['segment'].fillna(df['segment'].mode()[0])

# Feature engineering
df['month']         = df['date'].dt.month
df['quarter']       = df['date'].dt.quarter
df['month_name']    = df['date'].dt.strftime('%b')
df['revenue_net']   = df['revenue'] * (1 - df['discount_pct'])
df['revenue_band']  = pd.cut(df['revenue_net'],
                              bins=[0, 500, 1000, 1500, 9999],
                              labels=['Low', 'Mid', 'High', 'Premium'])

null_after = df.isnull().sum().sum()
print(f"Null values after  : {null_after}")
print(f"Null reduction     : {100 * (1 - null_after / df_raw.isnull().sum().sum()):.0f}%")
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — KPI CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: KPI CALCULATIONS")
print("=" * 60)

total_revenue       = df['revenue_net'].sum()
total_customers     = df['customer_id'].nunique()
avg_order_value     = df['revenue_net'].mean()
overall_churn_rate  = df['churned'].mean() * 100
total_units         = df['units_sold'].sum()
avg_discount        = df['discount_pct'].mean() * 100

# Customer Acquisition Rate (new customers per quarter)
cust_per_quarter = df.groupby('quarter')['customer_id'].nunique()
acq_rate = cust_per_quarter.mean()

# Revenue per segment
rev_per_segment = df.groupby('segment')['revenue_net'].sum().sort_values(ascending=False)

# Churn by segment
churn_by_segment = df.groupby('segment')['churned'].mean() * 100

print(f"Total Revenue       : ₹{total_revenue:,.0f}")
print(f"Total Customers     : {total_customers:,}")
print(f"Avg Order Value     : ₹{avg_order_value:,.0f}")
print(f"Overall Churn Rate  : {overall_churn_rate:.1f}%")
print(f"Total Units Sold    : {total_units:,}")
print(f"Avg Discount        : {avg_discount:.1f}%")
print(f"\nRevenue by Segment:\n{rev_per_segment.apply(lambda x: f'₹{x:,.0f}')}")
print(f"\nChurn by Segment:\n{churn_by_segment.round(1)}")
print()

# Identify underperforming products (bottom 2 by revenue)
rev_by_product = df.groupby('product')['revenue_net'].sum().sort_values()
underperforming = rev_by_product.head(2).index.tolist()
print(f"Underperforming products (bottom 2): {underperforming}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — A/B TEST: Discount vs No-Discount Conversion
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: A/B TEST — Discount vs Standard Pricing")
print("=" * 60)

group_a = df[df['discount_pct'] < 0.10]   # Standard pricing
group_b = df[df['discount_pct'] >= 0.10]  # Discount-led

conv_a = 1 - group_a['churned'].mean()
conv_b = 1 - group_b['churned'].mean()
lift   = (conv_b - conv_a) / conv_a * 100

t_stat, p_value = stats.ttest_ind(1 - group_a['churned'], 1 - group_b['churned'])
significance = "Statistically significant" if p_value < 0.05 else "Not significant"

print(f"Group A (Standard) conversion  : {conv_a*100:.1f}%  (n={len(group_a):,})")
print(f"Group B (Discount) conversion  : {conv_b*100:.1f}%  (n={len(group_b):,})")
print(f"Lift                           : +{lift:.1f}%")
print(f"p-value                        : {p_value:.4f} → {significance}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CHURN PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4: CHURN PREDICTION MODEL (Random Forest)")
print("=" * 60)

le = LabelEncoder()
df_model = df.copy()
for col in ['product', 'region', 'segment', 'channel', 'revenue_band']:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

features = ['product', 'region', 'segment', 'channel',
            'revenue_net', 'units_sold', 'discount_pct', 'month', 'quarter']
X = df_model[features]
y = df_model['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}")
print(f"Recall    : {rec:.3f}")
print(f"F1-Score  : {f1:.3f}")
print()

feat_imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — DASHBOARD VISUALIZATION (6 pages)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: GENERATING DASHBOARD...")
print("=" * 60)

# ── PAGE 1: KPI Overview ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Sales & Customer Insights Dashboard\nPage 1: KPI Overview',
             fontsize=16, fontweight='bold', color=BLUE, y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

kpis = [
    ('Total Revenue', f'₹{total_revenue/1e6:.1f}M', BLUE),
    ('Total Customers', f'{total_customers:,}', ACCENT),
    ('Avg Order Value', f'₹{avg_order_value:,.0f}', GREEN),
    ('Churn Rate', f'{overall_churn_rate:.1f}%', RED),
    ('Units Sold', f'{total_units:,}', ORANGE),
    ('Avg Discount', f'{avg_discount:.1f}%', GRAY),
    ('Underperforming', '2 products', RED),
    ('A/B Test Lift', f'+{lift:.0f}%', GREEN),
]

for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ax.set_facecolor('#FFFFFF')
    ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=22,
            fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=10,
            color=GRAY, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#E2E8F0')
        spine.set_linewidth(1.5)
        spine.set_visible(True)

plt.savefig('page1_kpi_overview.png', dpi=150, bbox_inches='tight',
            facecolor='#F8FAFC')
plt.close()
print("  ✓ Page 1 saved")

# ── PAGE 2: Revenue by Region & Product ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sales & Customer Insights Dashboard\nPage 2: Revenue by Region & Product',
             fontsize=16, fontweight='bold', color=BLUE)
fig.patch.set_facecolor('#F8FAFC')

rev_region = df.groupby('region')['revenue_net'].sum().sort_values(ascending=True)
bars = axes[0].barh(rev_region.index, rev_region.values / 1e6, color=PALETTE)
axes[0].set_xlabel('Revenue (₹ Millions)', color=GRAY)
axes[0].set_title('Revenue by Region', fontweight='bold', color=BLUE, pad=12)
for bar, val in zip(bars, rev_region.values):
    axes[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'₹{val/1e6:.1f}M', va='center', fontsize=9, color=GRAY)

rev_product = df.groupby('product')['revenue_net'].sum().sort_values(ascending=False)
colors = [RED if p in underperforming else ACCENT for p in rev_product.index]
bars2 = axes[1].bar(range(len(rev_product)), rev_product.values / 1e6, color=colors)
axes[1].set_xticks(range(len(rev_product)))
axes[1].set_xticklabels(rev_product.index, rotation=20, ha='right', fontsize=9)
axes[1].set_ylabel('Revenue (₹ Millions)', color=GRAY)
axes[1].set_title('Revenue by Product\n(Red = Underperforming)', fontweight='bold', color=BLUE, pad=12)
for bar, val in zip(bars2, rev_product.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'₹{val/1e6:.1f}M', ha='center', fontsize=9, color=GRAY)

plt.tight_layout()
plt.savefig('page2_revenue_region_product.png', dpi=150, bbox_inches='tight',
            facecolor='#F8FAFC')
plt.close()
print("  ✓ Page 2 saved")

# ── PAGE 3: Monthly Trend & Quarterly Performance ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sales & Customer Insights Dashboard\nPage 3: Revenue Trend & Quarterly Breakdown',
             fontsize=16, fontweight='bold', color=BLUE)
fig.patch.set_facecolor('#F8FAFC')

monthly = df.groupby('month')['revenue_net'].sum() / 1e6
months_label = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[0].plot(monthly.index, monthly.values, color=ACCENT, linewidth=2.5, marker='o',
             markersize=6, markerfacecolor=BLUE)
axes[0].fill_between(monthly.index, monthly.values, alpha=0.1, color=ACCENT)
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(months_label, fontsize=9)
axes[0].set_ylabel('Revenue (₹ Millions)', color=GRAY)
axes[0].set_title('Monthly Revenue Trend', fontweight='bold', color=BLUE, pad=12)

quarterly = df.groupby(['quarter', 'product'])['revenue_net'].sum().unstack() / 1e6
quarterly.plot(kind='bar', ax=axes[1], color=PALETTE, width=0.7)
axes[1].set_xticklabels([f'Q{q}' for q in quarterly.index], rotation=0)
axes[1].set_ylabel('Revenue (₹ Millions)', color=GRAY)
axes[1].set_title('Quarterly Revenue by Product', fontweight='bold', color=BLUE, pad=12)
axes[1].legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('page3_trend_quarterly.png', dpi=150, bbox_inches='tight',
            facecolor='#F8FAFC')
plt.close()
print("  ✓ Page 3 saved")

# ── PAGE 4: Customer Segmentation ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sales & Customer Insights Dashboard\nPage 4: Customer Segmentation & Churn',
             fontsize=16, fontweight='bold', color=BLUE)
fig.patch.set_facecolor('#F8FAFC')

seg_rev = df.groupby('segment')['revenue_net'].sum()
wedge_colors = [BLUE, ACCENT, GREEN, ORANGE, RED]
axes[0].pie(seg_rev.values, labels=seg_rev.index, autopct='%1.1f%%',
            colors=wedge_colors, startangle=90,
            textprops={'fontsize': 9},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Revenue by Customer Segment', fontweight='bold', color=BLUE, pad=12)

churn_seg = df.groupby('segment')['churned'].mean() * 100
colors_churn = [RED if v > 15 else ORANGE if v > 10 else GREEN for v in churn_seg.values]
bars = axes[1].bar(churn_seg.index, churn_seg.values, color=colors_churn, width=0.6)
axes[1].axhline(overall_churn_rate, color=GRAY, linestyle='--', linewidth=1.5,
                label=f'Avg {overall_churn_rate:.1f}%')
axes[1].set_ylabel('Churn Rate (%)', color=GRAY)
axes[1].set_title('Churn Rate by Segment\n(Red > 15%, Orange > 10%)',
                  fontweight='bold', color=BLUE, pad=12)
axes[1].legend(fontsize=9)
for bar, val in zip(bars, churn_seg.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=9, color=GRAY)

plt.tight_layout()
plt.savefig('page4_segmentation_churn.png', dpi=150, bbox_inches='tight',
            facecolor='#F8FAFC')
plt.close()
print("  ✓ Page 4 saved")

# ── PAGE 5: A/B Test Results ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sales & Customer Insights Dashboard\nPage 5: A/B Test — Discount vs Standard Pricing',
             fontsize=16, fontweight='bold', color=BLUE)
fig.patch.set_facecolor('#F8FAFC')

groups   = ['Group A\n(Standard)', 'Group B\n(Discount-led)']
conv_vals = [conv_a * 100, conv_b * 100]
bar_colors = [BLUE, GREEN]
bars = axes[0].bar(groups, conv_vals, color=bar_colors, width=0.5)
axes[0].set_ylabel('Conversion Rate (%)', color=GRAY)
axes[0].set_title(f'Conversion Rate Comparison\np-value: {p_value:.4f} ({significance})',
                  fontweight='bold', color=BLUE, pad=12)
axes[0].set_ylim(0, 100)
for bar, val in zip(bars, conv_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold', color=GRAY)
axes[0].annotate(f'+{lift:.0f}% lift', xy=(1, conv_b * 100),
                 xytext=(0.5, (conv_a + conv_b) * 50 + 5),
                 fontsize=11, color=GREEN, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=GREEN))

rev_a = group_a['revenue_net'].mean()
rev_b = group_b['revenue_net'].mean()
axes[1].bar(['Standard\nAvg Revenue', 'Discount\nAvg Revenue'], [rev_a, rev_b],
            color=[BLUE, ORANGE], width=0.5)
axes[1].set_ylabel('Avg Revenue per Order (₹)', color=GRAY)
axes[1].set_title('Avg Revenue: Standard vs Discount', fontweight='bold', color=BLUE, pad=12)
for i, val in enumerate([rev_a, rev_b]):
    axes[1].text(i, val + 5, f'₹{val:,.0f}', ha='center', fontsize=11,
                 fontweight='bold', color=GRAY)

plt.tight_layout()
plt.savefig('page5_ab_test.png', dpi=150, bbox_inches='tight', facecolor='#F8FAFC')
plt.close()
print("  ✓ Page 5 saved")

# ── PAGE 6: Model Evaluation ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sales & Customer Insights Dashboard\nPage 6: Churn Prediction Model Evaluation',
             fontsize=16, fontweight='bold', color=BLUE)
fig.patch.set_facecolor('#F8FAFC')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values  = [acc, prec, rec, f1]
colors_m = [GREEN if v >= 0.75 else ORANGE for v in values]
bars = axes[0].bar(metrics, values, color=colors_m, width=0.5)
axes[0].set_ylim(0, 1.1)
axes[0].axhline(0.75, color=GRAY, linestyle='--', linewidth=1, label='Target (0.75)')
axes[0].set_ylabel('Score', color=GRAY)
axes[0].set_title('Model Performance Metrics\n(Random Forest Classifier)',
                  fontweight='bold', color=BLUE, pad=12)
axes[0].legend(fontsize=9)
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold', color=GRAY)

feat_top = feat_imp.head(6)
axes[1].barh(feat_top.index[::-1], feat_top.values[::-1], color=ACCENT)
axes[1].set_xlabel('Feature Importance', color=GRAY)
axes[1].set_title('Top 6 Features for Churn Prediction',
                  fontweight='bold', color=BLUE, pad=12)
for i, (val, name) in enumerate(zip(feat_top.values[::-1], feat_top.index[::-1])):
    axes[1].text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9, color=GRAY)

plt.tight_layout()
plt.savefig('page6_model_evaluation.png', dpi=150, bbox_inches='tight',
            facecolor='#F8FAFC')
plt.close()
print("  ✓ Page 6 saved")

print()
print("=" * 60)
print("ALL 6 DASHBOARD PAGES GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\nFiles saved:")
for i in range(1, 7):
    print(f"  page{i}_*.png")
