# 🛡️ Insurance Risk Modeling & IFRS 17 Analytical Framework

![R](https://img.shields.io/badge/Language-R-276DC3?style=flat&logo=r)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Actuarial%20Analytics-purple)
![Standard](https://img.shields.io/badge/Standard-IFRS%2017-darkblue)

## 📌 Project Overview

This project builds an **actuarial-oriented analytical framework** for insurance portfolio risk modeling under the **IFRS 17** accounting standard. It simulates an insurance portfolio, analyzes claim risk patterns, models reserve behavior, and produces financial reporting summaries aligned with IFRS 17 principles including the **Building Blocks Approach (BBA)**.

---

## 🎯 Business Objectives

- Simulate and analyze insurance portfolio metrics under IFRS 17
- Model claim frequency, severity, and reserve projections
- Identify high-risk policy segments for underwriting optimization
- Produce actuarial summary reports for financial reporting compliance

---

## 🗂️ Dataset Description

| Variable | Description |
|---|---|
| `policy_id` | Unique policy identifier |
| `product_type` | Auto / Health / Property / Life |
| `premium_amount` | Annual premium (CAD) |
| `policy_duration` | Policy term in months |
| `claim_flag` | Binary: 1 = claim filed, 0 = no claim |
| `claim_amount` | Claim value if applicable |
| `insured_age` | Age of policyholder |
| `risk_class` | Underwriting risk category (1–5) |
| `region` | Geographic region |
| `inception_date` | Policy start date |

> ⚠️ *All data is simulated. No real policyholder data was used.*

---

## 🔧 Methods & Tools

### Risk Modeling
- **Claim Frequency Model** — Poisson regression for claim occurrence
- **Claim Severity Model** — Gamma regression for claim amounts
- **Pure Premium** — Combined frequency × severity approach
- **Risk Classification** — Clustering of policy segments by risk profile

### Reserve Modeling (IFRS 17)
- **Chain-Ladder Method** — Claims development triangle analysis
- **Bornhuetter-Ferguson Method** — Reserve estimation with prior information
- **Liability for Remaining Coverage (LRC)** — IFRS 17 BBA component
- **Contractual Service Margin (CSM)** — Unearned profit tracking

### Reporting
- Loss ratio, Combined ratio, Reserve adequacy metrics
- Actuarial summary tables by product line and cohort

### Tools
```
R | tidyverse | ChainLadder | actuar | ggplot2 | knitr | dplyr | lubridate
```

---

## 📊 Key Results

### Portfolio Summary (Simulated — 8,000 policies)

| Metric | Value |
|---|---|
| Total Gross Written Premium | CAD 18.4M |
| Overall Claim Frequency | 12.7% |
| Average Claim Severity | CAD 6,840 |
| Loss Ratio | 68.3% |
| Combined Ratio | 91.4% |
| Total IBNR Reserve | CAD 2.1M |

### Risk Segment Analysis

| Risk Class | Claim Frequency | Avg. Severity | Loss Ratio |
|---|---|---|---|
| Class 1 (Low) | 4.2% | CAD 2,100 | 38.5% |
| Class 2 | 8.7% | CAD 4,300 | 57.2% |
| Class 3 | 13.4% | CAD 6,800 | 72.1% |
| Class 4 | 21.8% | CAD 9,200 | 88.6% |
| Class 5 (High) | 34.1% | CAD 15,400 | 118.3% |

> ⚠️ Class 5 policies are **loss-making** — underwriting action recommended

---

## 📁 Project Structure

```
insurance-risk-ifrs17/
│
├── data/
│   └── simulated_insurance_portfolio.csv
├── scripts/
│   ├── 01_data_simulation.R
│   ├── 02_claim_frequency_model.R
│   ├── 03_claim_severity_model.R
│   ├── 04_reserve_modeling.R
│   ├── 05_ifrs17_framework.R
│   └── 06_actuarial_report.R
├── outputs/
│   ├── loss_ratio_by_segment.png
│   ├── claims_development_triangle.png
│   ├── reserve_projection.png
│   └── ifrs17_summary_report.pdf
└── README.md
```

---

## 💡 Key Actuarial Insights

1. **Class 5 policies generate a combined ratio of 118%** — immediate re-pricing or exclusion needed
2. **Health product line** shows the highest IBNR uncertainty (development tail > 36 months)
3. **Younger policyholders (18–30)** show 2.4x higher claim frequency in auto segment
4. **CSM erosion rate** is highest in Q3 due to seasonal claim spikes (storm-related property claims)
5. Reserve adequacy at **98.7% confidence** under current Chain-Ladder estimates

---

## 📚 Concepts Applied

- IFRS 17 Building Blocks Approach (BBA)
- Liability for Remaining Coverage (LRC)
- Liability for Incurred Claims (LIC)
- Contractual Service Margin (CSM)
- IBNR (Incurred But Not Reported) reserves
- Chain-Ladder & Bornhuetter-Ferguson methods
- GLM-based actuarial pricing models

---

## 👤 Author

**Adededji Djamiou ABAYOMI**
Data Analyst | Quantitative Modeling | Business Intelligence
📍 Montréal, QC, Canada
📧 abayomi.adededji.djamiou@gmail.com
🔗 [LinkedIn](https://linkedin.com)
