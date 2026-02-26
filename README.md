# Credit_Tool

# Contact Info

Want to hire me? Check out my LinkedIn here: https://www.linkedin.com/in/jesse-o-03476a102/

Want to comission me for a project? Check out my Upwork profile here: https://www.upwork.com/freelancers/~0193f57dd84700cb81

# CreditForge
A Lightweight Credit Underwriting & Stress Testing Engine (Flask + Python)

CreditForge is a simple underwriting application that automates core loan analysis workflows.

It transforms borrower financial data and loan terms into:

Repayment capacity analysis

Safety rule (covenant) checks

Stress testing under adverse scenarios

Auto-generated PDF credit memo

# What Problem Does This Solve?

In traditional lending workflows, analysts manually:

Calculate repayment strength

Compare financial health against loan safety thresholds

Test downside scenarios

Draft summary memos

CreditForge automates this into a repeatable, transparent decision process.

# What It Does
1. Cash Flow Analysis

Estimates available cash for debt repayment.

2. Annual Payment Calculation

Computes yearly loan payment requirements.

3. Financial Strength Metrics

Measures:

Repayment coverage

Profit strength relative to debt

Collateral protection

4. Covenant Monitoring

Checks whether predefined loan safety rules are satisfied and reports headroom.

5. Stress Testing

Simulates:

Rising interest rates

Declining profits

Combined downside scenarios

6. PDF Credit Memo Export

Generates a structured summary suitable for internal review.

# Tech Stack

Python

Flask

Pandas

NumPy

ReportLab (PDF generation)

# Setup
pip install -r requirements.txt
python app.py

Then open:

http://localhost:55000
📂 Required CSV Structure
Financials (Single Row)

# Required columns:

period
revenues
ebitda
ebit
interest_expense
capex
cash_taxes
working_capital_change
total_assets
total_liabilities
Debt Terms (Single Row)

# Required columns:

facility_name
loan_balance
interest_rate
amortization_years
collateral_value
# Design Philosophy

Transparent math

No black-box modeling

Fast triage decision support

Workflow automation over complexity

This is intentionally simplified for demonstration and prototyping purposes.

# Disclaimer

This tool uses simplified financial assumptions and should not replace full underwriting models, professional judgment, or institutional policy.

# Potential Future Enhancements

Multi-period trend analysis

Portfolio-level monitoring dashboard

Early warning risk scoring

# Author

Jesse Olivarez | Finance | Credit Risk | Data Analytics
Monte Carlo scenario simulation

API deployment for SaaS integration
