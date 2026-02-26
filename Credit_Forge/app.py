import io
import os
import json
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, send_file, render_template_string

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB uploads

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


# ----------------------------
# Finance logic
# ----------------------------

@dataclass
class Covenants:
    min_dscr: float = 1.25
    min_interest_coverage: float = 2.00
    max_leverage: float = 4.00
    max_ltv: float = 0.75


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_one_row_csv(file_storage, required_cols: List[str]) -> pd.Series:
    df = pd.read_csv(file_storage)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) < 1:
        raise ValueError("CSV has no rows.")
    return df.iloc[0]


def compute_metrics(fin: pd.Series, debt: pd.Series) -> Dict[str, Any]:
    # Financials
    revenues = safe_float(fin["revenues"])
    ebitda = safe_float(fin["ebitda"])
    ebit = safe_float(fin["ebit"])
    interest_expense = max(safe_float(fin["interest_expense"]), 0.0)
    capex = max(safe_float(fin["capex"]), 0.0)
    cash_taxes = max(safe_float(fin["cash_taxes"]), 0.0)
    wc_change = safe_float(fin["working_capital_change"])  # could be +/- (use sign)
    total_assets = safe_float(fin["total_assets"])
    total_liabilities = safe_float(fin["total_liabilities"])

    # Debt terms
    loan_balance = max(safe_float(debt["loan_balance"]), 0.0)
    interest_rate = max(safe_float(debt["interest_rate"]), 0.0)
    amort_years = max(safe_float(debt["amortization_years"]), 1.0)
    collateral_value = max(safe_float(debt["collateral_value"]), 1.0)

    # Simple annual debt service proxy: interest + straight-line principal
    annual_interest = loan_balance * interest_rate
    annual_principal = loan_balance / amort_years
    annual_debt_service = annual_interest + annual_principal

    # Cash flow proxy (simple): EBITDA - capex - cash taxes - ΔWC (if WC increases, cash outflow)
    # working_capital_change positive = cash outflow
    cash_flow_available = ebitda - capex - cash_taxes - wc_change

    # Key ratios
    dscr = cash_flow_available / annual_debt_service if annual_debt_service > 0 else np.inf
    interest_coverage = ebit / interest_expense if interest_expense > 0 else np.inf
    leverage = loan_balance / ebitda if ebitda > 0 else np.inf
    ltv = loan_balance / collateral_value if collateral_value > 0 else np.inf

    # Balance sheet quick sanity
    equity = total_assets - total_liabilities

    return {
        "revenues": revenues,
        "ebitda": ebitda,
        "ebit": ebit,
        "cash_flow_available": cash_flow_available,
        "loan_balance": loan_balance,
        "interest_rate": interest_rate,
        "annual_interest": annual_interest,
        "annual_principal": annual_principal,
        "annual_debt_service": annual_debt_service,
        "dscr": dscr,
        "interest_coverage": interest_coverage,
        "leverage": leverage,
        "ltv": ltv,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "equity": equity,
        "facility_name": str(debt.get("facility_name", "Facility")),
        "period": str(fin.get("period", "LTM")),
    }


def covenant_check(metrics: Dict[str, Any], cov: Covenants) -> Dict[str, Any]:
    def headroom_min(actual, minimum):
        return actual - minimum

    def headroom_max(actual, maximum):
        return maximum - actual

    checks = {
        "DSCR (min)": {
            "actual": metrics["dscr"],
            "threshold": cov.min_dscr,
            "pass": metrics["dscr"] >= cov.min_dscr,
            "headroom": headroom_min(metrics["dscr"], cov.min_dscr),
        },
        "Interest Coverage (min)": {
            "actual": metrics["interest_coverage"],
            "threshold": cov.min_interest_coverage,
            "pass": metrics["interest_coverage"] >= cov.min_interest_coverage,
            "headroom": headroom_min(metrics["interest_coverage"], cov.min_interest_coverage),
        },
        "Leverage (max)": {
            "actual": metrics["leverage"],
            "threshold": cov.max_leverage,
            "pass": metrics["leverage"] <= cov.max_leverage,
            "headroom": headroom_max(metrics["leverage"], cov.max_leverage),
        },
        "LTV (max)": {
            "actual": metrics["ltv"],
            "threshold": cov.max_ltv,
            "pass": metrics["ltv"] <= cov.max_ltv,
            "headroom": headroom_max(metrics["ltv"], cov.max_ltv),
        },
    }
    overall_pass = all(v["pass"] for v in checks.values())
    return {"overall_pass": overall_pass, "checks": checks}


def stress_test(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Stress grid:
      - rate shock: +0%, +1%, +2%, +3%
      - ebitda shock: 0%, -10%, -20%, -30%
    Recompute DSCR & Leverage under each scenario.
    """
    base_balance = metrics["loan_balance"]
    base_rate = metrics["interest_rate"]
    amort_years = max(metrics["loan_balance"] / max(metrics["annual_principal"], 1.0), 1.0)

    base_ebitda = max(metrics["ebitda"], 0.0)
    base_cfa = metrics["cash_flow_available"]

    rate_shocks = [0.00, 0.01, 0.02, 0.03]
    ebitda_shocks = [0.00, -0.10, -0.20, -0.30]

    rows = []
    for rs in rate_shocks:
        for es in ebitda_shocks:
            new_rate = base_rate + rs
            annual_interest = base_balance * new_rate
            annual_principal = base_balance / amort_years
            debt_service = annual_interest + annual_principal

            # Cash flow available: scale EBITDA shock, keep other items embedded in CFA approximation
            # We approximate by shifting CFA by delta EBITDA (since CFA = EBITDA - other stuff)
            shocked_ebitda = base_ebitda * (1.0 + es)
            delta_ebitda = shocked_ebitda - base_ebitda
            shocked_cfa = base_cfa + delta_ebitda

            dscr = shocked_cfa / debt_service if debt_service > 0 else np.inf
            leverage = base_balance / shocked_ebitda if shocked_ebitda > 0 else np.inf

            rows.append({
                "Rate Shock": f"+{int(rs*100)}%",
                "EBITDA Shock": f"{int(es*100)}%",
                "Shocked Rate": new_rate,
                "Shocked EBITDA": shocked_ebitda,
                "DSCR": dscr,
                "Leverage": leverage,
            })

    return pd.DataFrame(rows)


# ----------------------------
# PDF generation
# ----------------------------

def money(x: float) -> str:
    if np.isinf(x):
        return "∞"
    return f"${x:,.0f}"


def pct(x: float) -> str:
    if np.isinf(x):
        return "∞"
    return f"{x*100:,.2f}%"


def ratio(x: float) -> str:
    if np.isinf(x):
        return "∞"
    return f"{x:,.2f}x"


def build_credit_memo_pdf(run_id: str, payload: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    x0 = 0.75 * inch
    y = height - 0.85 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x0, y, "CreditForge — Auto-Generated Credit Memo")
    y -= 0.25 * inch

    c.setFont("Helvetica", 10)
    c.drawString(x0, y, f"Run ID: {run_id}")
    y -= 0.18 * inch

    metrics = payload["metrics"]
    cov = payload["covenants"]
    stress = pd.DataFrame(payload["stress_table"])

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "1) Snapshot")
    y -= 0.20 * inch

    c.setFont("Helvetica", 10)
    snapshot_lines = [
        f"Period: {metrics['period']}",
        f"Facility: {metrics['facility_name']}",
        f"Loan Balance: {money(metrics['loan_balance'])}",
        f"Rate: {pct(metrics['interest_rate'])}",
        f"Revenues: {money(metrics['revenues'])}",
        f"EBITDA: {money(metrics['ebitda'])}",
        f"Cash Flow Available (proxy): {money(metrics['cash_flow_available'])}",
    ]
    for line in snapshot_lines:
        c.drawString(x0, y, line)
        y -= 0.16 * inch

    y -= 0.10 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "2) Key Ratios")
    y -= 0.20 * inch
    c.setFont("Helvetica", 10)

    ratios_lines = [
        f"DSCR: {ratio(metrics['dscr'])}",
        f"Interest Coverage: {ratio(metrics['interest_coverage'])}",
        f"Leverage: {ratio(metrics['leverage'])}",
        f"LTV: {pct(metrics['ltv'])}",
    ]
    for line in ratios_lines:
        c.drawString(x0, y, line)
        y -= 0.16 * inch

    y -= 0.10 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "3) Covenant Check (Pass/Fail)")
    y -= 0.20 * inch

    c.setFont("Helvetica", 10)
    overall = "PASS" if cov["overall_pass"] else "FAIL"
    c.drawString(x0, y, f"Overall: {overall}")
    y -= 0.18 * inch

    for name, v in cov["checks"].items():
        status = "PASS" if v["pass"] else "FAIL"
        c.drawString(
            x0, y,
            f"{name}: {ratio(v['actual']) if 'LTV' not in name else pct(v['actual'])} "
            f"vs {ratio(v['threshold']) if 'LTV' not in name else pct(v['threshold'])} "
            f"→ {status} (Headroom: {v['headroom']:.2f})"
        )
        y -= 0.16 * inch

    # New page for stress table
    c.showPage()
    y = height - 0.85 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "4) Stress Test Grid (Rate ↑, EBITDA ↓)")
    y -= 0.25 * inch

    c.setFont("Helvetica", 9)
    c.drawString(x0, y, "Showing DSCR + Leverage under scenario shocks (quick screening, not a full model).")
    y -= 0.25 * inch

    # Render a small table (first 16 rows is enough for 4x4)
    cols = ["Rate Shock", "EBITDA Shock", "DSCR", "Leverage"]
    stress_small = stress[cols].copy().head(16)

    # Table header
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0, y, " | ".join(cols))
    y -= 0.16 * inch
    c.setFont("Helvetica", 9)

    for _, row in stress_small.iterrows():
        line = f"{row['Rate Shock']} | {row['EBITDA Shock']} | {row['DSCR']:.2f}x | {row['Leverage']:.2f}x"
        c.drawString(x0, y, line)
        y -= 0.14 * inch
        if y < 0.9 * inch:
            c.showPage()
            y = height - 0.85 * inch
            c.setFont("Helvetica", 9)

    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, height - 0.85 * inch, "Notes / Disclosures")
    c.setFont("Helvetica", 9)
    c.drawString(x0, height - 1.10 * inch, "This memo is autogenerated from uploaded CSV inputs and simplified assumptions.")
    c.drawString(x0, height - 1.26 * inch, "Use for workflow acceleration + triage, not as a substitute for full underwriting.")
    c.save()

    buf.seek(0)
    return buf.read()


# ----------------------------
# Flask UI
# ----------------------------

HOME_HTML = """
<!doctype html>
<html>
<head>
  <title>CreditForge</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 900px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-top: 16px; }
    .muted { color: #666; }
    input[type=file] { margin-top: 8px; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #222; background: #111; color: #fff; cursor: pointer; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>CreditForge</h1>
  <p class="muted">Upload borrower financials + debt terms → instant ratios, covenant checks, stress tests, and a PDF credit memo.</p>

  <div class="card">
    <h3>Upload</h3>
    <form method="POST" action="/analyze" enctype="multipart/form-data">
      <div>
        <label><b>Financials CSV</b> (one row). Required columns:</label>
        <div class="muted"><code>period, revenues, ebitda, ebit, interest_expense, capex, cash_taxes, working_capital_change, total_assets, total_liabilities</code></div>
        <input type="file" name="financials" accept=".csv" required>
      </div>
      <div style="margin-top:14px;">
        <label><b>Debt Terms CSV</b> (one row). Required columns:</label>
        <div class="muted"><code>facility_name, loan_balance, interest_rate, amortization_years, collateral_value</code></div>
        <input type="file" name="debt_terms" accept=".csv" required>
      </div>

      <div style="margin-top:14px;">
        <label><b>Covenants</b> (optional overrides)</label><br/>
        Min DSCR: <input name="min_dscr" placeholder="1.25" style="width:80px;">
        Min Int Cov: <input name="min_intcov" placeholder="2.00" style="width:80px;">
        Max Leverage: <input name="max_lev" placeholder="4.00" style="width:80px;">
        Max LTV: <input name="max_ltv" placeholder="0.75" style="width:80px;">
      </div>

      <div style="margin-top:16px;">
        <button type="submit">Analyze</button>
      </div>
    </form>
  </div>

  <div class="card">
    <h3>Quick start</h3>
    <ol>
      <li>Put this folder on your machine</li>
      <li><code>pip install -r requirements.txt</code></li>
      <li><code>python app.py</code></li>
      <li>Open <code>http://127.0.0.1:5055</code></li>
      <li>Use <code>sample_financials.csv</code> and <code>sample_debt_terms.csv</code></li>
    </ol>
  </div>
</body>
</html>
"""

RESULTS_HTML = """
<!doctype html>
<html>
<head>
  <title>CreditForge Results</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 1100px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; border-bottom: 1px solid #eee; text-align: left; font-size: 14px; }
    .pass { color: #0a7; font-weight: bold; }
    .fail { color: #c22; font-weight: bold; }
    a.button { display:inline-block; padding: 10px 14px; border-radius: 10px; border: 1px solid #222; background:#111; color:#fff; text-decoration:none; }
    .muted { color:#666; }
    code { background:#f6f6f6; padding: 2px 6px; border-radius:6px; }
  </style>
</head>
<body>
  <h1>Results</h1>
  <p class="muted">Run ID: <code>{{ run_id }}</code></p>

  <div style="margin: 12px 0;">
    <a class="button" href="/memo/{{ run_id }}">Download PDF Credit Memo</a>
    <a style="margin-left:10px;" href="/">New run</a>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Snapshot</h3>
      <table>
        <tr><th>Facility</th><td>{{ metrics.facility_name }}</td></tr>
        <tr><th>Period</th><td>{{ metrics.period }}</td></tr>
        <tr><th>Loan Balance</th><td>{{ money(metrics.loan_balance) }}</td></tr>
        <tr><th>Rate</th><td>{{ pct(metrics.interest_rate) }}</td></tr>
        <tr><th>Revenues</th><td>{{ money(metrics.revenues) }}</td></tr>
        <tr><th>EBITDA</th><td>{{ money(metrics.ebitda) }}</td></tr>
        <tr><th>Cash Flow Available (proxy)</th><td>{{ money(metrics.cash_flow_available) }}</td></tr>
      </table>
    </div>

    <div class="card">
      <h3>Key Ratios</h3>
      <table>
        <tr><th>DSCR</th><td>{{ ratio(metrics.dscr) }}</td></tr>
        <tr><th>Interest Coverage</th><td>{{ ratio(metrics.interest_coverage) }}</td></tr>
        <tr><th>Leverage</th><td>{{ ratio(metrics.leverage) }}</td></tr>
        <tr><th>LTV</th><td>{{ pct(metrics.ltv) }}</td></tr>
      </table>
    </div>

    <div class="card" style="grid-column: 1 / span 2;">
      <h3>Covenant Check</h3>
      <p>Overall:
        {% if cov.overall_pass %}
          <span class="pass">PASS</span>
        {% else %}
          <span class="fail">FAIL</span>
        {% endif %}
      </p>
      <table>
        <tr><th>Covenant</th><th>Actual</th><th>Threshold</th><th>Status</th><th>Headroom</th></tr>
        {% for name, v in cov.checks.items() %}
          <tr>
            <td>{{ name }}</td>
            <td>
              {% if "LTV" in name %}
                {{ pct(v.actual) }}
              {% else %}
                {{ ratio(v.actual) }}
              {% endif %}
            </td>
            <td>
              {% if "LTV" in name %}
                {{ pct(v.threshold) }}
              {% else %}
                {{ ratio(v.threshold) }}
              {% endif %}
            </td>
            <td>
              {% if v.pass %}
                <span class="pass">PASS</span>
              {% else %}
                <span class="fail">FAIL</span>
              {% endif %}
            </td>
            <td>{{ "%.2f"|format(v.headroom) }}</td>
          </tr>
        {% endfor %}
      </table>
    </div>

    <div class="card" style="grid-column: 1 / span 2;">
      <h3>Stress Test Grid (first 16 scenarios)</h3>
      <p class="muted">Rate shocks (+0 to +3%) and EBITDA shocks (0 to -30%). This is a fast triage view.</p>
      <table>
        <tr><th>Rate Shock</th><th>EBITDA Shock</th><th>DSCR</th><th>Leverage</th></tr>
        {% for row in stress %}
          <tr>
            <td>{{ row["Rate Shock"] }}</td>
            <td>{{ row["EBITDA Shock"] }}</td>
            <td>{{ "%.2f"|format(row["DSCR"]) }}x</td>
            <td>{{ "%.2f"|format(row["Leverage"]) }}x</td>
          </tr>
        {% endfor %}
      </table>
    </div>
  </div>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "financials" not in request.files or "debt_terms" not in request.files:
        return "Missing files.", 400

    fin_file = request.files["financials"]
    debt_file = request.files["debt_terms"]

    fin_required = [
        "period", "revenues", "ebitda", "ebit", "interest_expense",
        "capex", "cash_taxes", "working_capital_change",
        "total_assets", "total_liabilities"
    ]
    debt_required = ["facility_name", "loan_balance", "interest_rate", "amortization_years", "collateral_value"]

    try:
        fin = load_one_row_csv(fin_file, fin_required)
        debt = load_one_row_csv(debt_file, debt_required)
    except Exception as e:
        return f"CSV error: {e}", 400

    cov = Covenants(
        min_dscr=safe_float(request.form.get("min_dscr"), 1.25) or 1.25,
        min_interest_coverage=safe_float(request.form.get("min_intcov"), 2.00) or 2.00,
        max_leverage=safe_float(request.form.get("max_lev"), 4.00) or 4.00,
        max_ltv=safe_float(request.form.get("max_ltv"), 0.75) or 0.75,
    )

    metrics = compute_metrics(fin, debt)
    cov_result = covenant_check(metrics, cov)
    stress = stress_test(metrics)

    run_id = uuid.uuid4().hex[:10]
    run_path = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_path, exist_ok=True)

    payload = {
        "metrics": metrics,
        "covenants": cov_result,
        "stress_table": stress.to_dict(orient="records"),
    }
    with open(os.path.join(run_path, "payload.json"), "w") as f:
        json.dump(payload, f, indent=2)

    return redirect(url_for("results", run_id=run_id))


@app.route("/results/<run_id>", methods=["GET"])
def results(run_id: str):
    run_path = os.path.join(RUNS_DIR, run_id, "payload.json")
    if not os.path.exists(run_path):
        return "Run not found.", 404

    with open(run_path, "r") as f:
        payload = json.load(f)

    metrics = payload["metrics"]
    cov = payload["covenants"]
    stress_rows = payload["stress_table"][:16]

    return render_template_string(
        RESULTS_HTML,
        run_id=run_id,
        metrics=metrics,
        cov=cov,
        stress=stress_rows,
        money=money,
        pct=pct,
        ratio=ratio,
    )


@app.route("/memo/<run_id>", methods=["GET"])
def memo(run_id: str):
    run_path = os.path.join(RUNS_DIR, run_id, "payload.json")
    if not os.path.exists(run_path):
        return "Run not found.", 404

    with open(run_path, "r") as f:
        payload = json.load(f)

    pdf_bytes = build_credit_memo_pdf(run_id, payload)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"CreditForge_Memo_{run_id}.pdf",
    )


if __name__ == "__main__":
    app.run(port=55000, debug=True, use_reloader=False)