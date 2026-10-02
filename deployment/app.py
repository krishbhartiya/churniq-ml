from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ──────────────────────────────────────────────────────────────────────────────
# Page / theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHURNIQ — Telecom Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #050505;
  --surface: #0b0b0c;
  --surface-2: #111113;
  --surface-3: #171719;
  --border: #252528;
  --border-soft: #1b1b1e;
  --text: #f5f5f5;
  --muted: #9b9b9f;
  --dim: #69696e;
  --accent: #ffffff;
  --blue: #7aa2ff;
  --green: #43d19e;
  --red: #ff6b7a;
  --amber: #e7b75b;
}

html, body, [class*="css"] {
  font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
  background:
    radial-gradient(circle at 50% -15%, rgba(255,255,255,.055), transparent 32%),
    radial-gradient(circle at 0% 50%, rgba(122,162,255,.035), transparent 25%),
    var(--bg);
  color: var(--text);
}

/* Hide Streamlit chrome while preserving controls */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

h1, h2, h3, h4 {
  font-family: "Space Grotesk", "DM Sans", sans-serif !important;
  letter-spacing: -0.035em;
  color: var(--text) !important;
}
h1 {font-size: 2.45rem !important; line-height: 1.08 !important; margin: .15rem 0 .65rem !important;}
h2 {font-size: 1.65rem !important;}
h3 {font-size: 1.08rem !important;}
p, li {color: #c2c2c6;}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #080809;
  border-right: 1px solid var(--border-soft);
}
section[data-testid="stSidebar"] > div {
  padding: 24px 16px 22px 16px;
}
.sidebar-brand {padding: 2px 6px 22px; border-bottom: 1px solid var(--border-soft); margin-bottom: 20px;}
.sidebar-brand .mark {font-family:"Space Grotesk"; font-size:1.35rem; font-weight:700; letter-spacing:-.05em;}
.sidebar-brand .mark span {color:#a7a7ac;}
.sidebar-brand .caption {font-size:.67rem; color:var(--dim); letter-spacing:.16em; text-transform:uppercase; margin-top:5px;}
.sidebar-section {font-size:.64rem; color:#66666b; letter-spacing:.16em; text-transform:uppercase; font-weight:700; margin:18px 6px 8px;}
.sidebar-note {font-size:.74rem; line-height:1.55; color:#77777d; margin:12px 6px;}

/* Native controls */
.stRadio > label, .stSelectbox > label, .stSlider > label, .stNumberInput > label,
.stTextInput > label, .stMultiSelect > label {color:#a8a8ad !important; font-size:.78rem !important; font-weight:600 !important;}
.stRadio [role="radiogroup"] {gap: 7px !important;}
.stRadio [role="radio"] {padding: 7px 10px !important; border:1px solid #222226; border-radius:10px; background:#0e0e10;}
.stRadio [role="radio"][aria-checked="true"] {border-color:#55555c; background:#18181b;}
.stButton > button, .stFormSubmitButton > button {
  border-radius: 11px !important;
  border: 1px solid #2c2c30 !important;
  background: #f4f4f5 !important;
  color: #080808 !important;
  font-weight: 700 !important;
  min-height: 42px;
  transition: all .18s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {transform:translateY(-1px); background:#ffffff !important; border-color:#555 !important;}

/* Layout primitives */
.page-top {display:flex; justify-content:space-between; align-items:flex-end; gap:18px; margin-bottom:18px;}
.eyebrow {color:#85858a; font-size:.66rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; margin-bottom:7px;}
.lead {max-width:880px; color:#a3a3a8; font-size:.93rem; line-height:1.7; margin-top:0;}
.hero {
  padding: 28px 30px;
  border:1px solid #29292d;
  border-radius:20px;
  background:linear-gradient(135deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
  box-shadow:0 24px 70px rgba(0,0,0,.28);
  margin-bottom:18px;
}
.hero .hero-title {font-family:"Space Grotesk"; font-size:2.35rem; font-weight:700; letter-spacing:-.055em; line-height:1.05; margin:7px 0 10px;}
.hero .hero-title span {color:#9a9aa0;}
.hero .hero-sub {max-width:780px; color:#9d9da2; line-height:1.65; font-size:.92rem;}

.card {background:linear-gradient(180deg,#101012,#0b0b0d); border:1px solid var(--border); border-radius:15px; padding:17px; height:100%;}
.card:hover {border-color:#303035;}
.metric-card {background:#0d0d0f; border:1px solid #232327; border-radius:14px; padding:14px 15px; min-height:90px;}
.metric-label {font-size:.62rem; color:#737379; text-transform:uppercase; letter-spacing:.14em; font-weight:700;}
.metric-value {font-family:"Space Grotesk"; font-size:1.42rem; font-weight:700; letter-spacing:-.035em; margin-top:7px; color:#f4f4f5; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.metric-sub {font-size:.68rem; color:#66666c; margin-top:4px;}
.section-head {display:flex; align-items:flex-end; justify-content:space-between; gap:12px; margin:22px 0 9px;}
.section-head .kicker {font-size:.62rem; color:#69696e; text-transform:uppercase; letter-spacing:.16em; font-weight:700;}
.section-head .title {font-family:"Space Grotesk"; font-size:1.15rem; font-weight:600; letter-spacing:-.03em;}
.muted {color:#77777d; font-size:.77rem; line-height:1.65;}
.small {color:#99999f; font-size:.78rem; line-height:1.65;}
.divider {height:1px; background:#1d1d20; margin:18px 0;}
.badge {display:inline-flex; align-items:center; gap:7px; padding:5px 9px; border:1px solid #2b2b30; border-radius:999px; color:#aaaab0; font-size:.66rem; font-weight:700;}
.dot {width:6px; height:6px; border-radius:50%; background:#d9d9dc; display:inline-block;}

/* Risk */
.risk-panel {border-radius:18px; padding:22px; border:1px solid #303035; background:linear-gradient(135deg,#111113,#0a0a0b);}
.risk-panel.high {border-color:rgba(255,107,122,.42); background:linear-gradient(135deg,rgba(255,107,122,.11),rgba(17,17,19,.96));}
.risk-panel.low {border-color:rgba(67,209,158,.35); background:linear-gradient(135deg,rgba(67,209,158,.09),rgba(17,17,19,.96));}
.risk-panel.watch {border-color:rgba(231,183,91,.38); background:linear-gradient(135deg,rgba(231,183,91,.09),rgba(17,17,19,.96));}
.risk-word {font-family:"Space Grotesk"; font-size:2.7rem; font-weight:700; letter-spacing:-.06em; margin-top:5px;}
.risk-caption {font-size:.76rem; color:#929298; line-height:1.6;}
.explain {border:1px solid #25252a; background:#0d0d0f; border-radius:13px; padding:15px;}
.signal {display:flex; gap:10px; align-items:flex-start; padding:10px 0; border-bottom:1px solid #1d1d20; font-size:.79rem; color:#b5b5ba;}
.signal:last-child {border-bottom:0;}
.signal .bullet {width:7px; height:7px; border-radius:50%; background:#b9b9be; margin-top:6px; flex:0 0 auto;}

/* Dataframes */
div[data-testid="stDataFrame"] {border:1px solid #242428; border-radius:12px; overflow:hidden;}
div[data-testid="stMetric"] {background:#0d0d0f; border:1px solid #232327; border-radius:12px; padding:10px 12px;}

/* Plotly container polish */
.js-plotly-plot .plotly {border-radius:12px;}

/* Expander */
details {background:#0d0d0f !important; border:1px solid #222226 !important; border-radius:12px !important;}

@media (max-width: 900px) {
  h1 {font-size:2rem !important;}
  .hero .hero-title {font-size:1.9rem;}
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Data / ML
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if DATA.exists():
        df = pd.read_csv(DATA)
    else:
        try:
            df = pd.read_csv(URL)
        except Exception as exc:
            st.error("The IBM Telco CSV could not be downloaded automatically.")
            st.code(f"{exc}")
            st.info("Place Telco-Customer-Churn.csv inside the project's data/ folder and restart Streamlit.")
            st.stop()
    required = {"customerID", "Churn", "TotalCharges", "tenure", "MonthlyCharges"}
    missing = sorted(required - set(df.columns))
    if missing:
        st.error("The dataset is missing required columns: " + ", ".join(missing))
        st.stop()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(DATA, index=False)
    except Exception:
        pass
    return df


def make_onehot_encoder():
    # sparse_output=False is required because GradientBoostingClassifier expects dense input.
    # The fallback keeps the project compatible with older scikit-learn releases.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_resource(show_spinner=False)
def build_models(df):
    X = df.drop(columns=["customerID", "Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [c for c in X.columns if c not in numeric]

    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]), categorical),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=.05, max_depth=3, random_state=42),
    }

    Xt, Xv, yt, yv = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    pipes, rows, cms = {}, {}, {}

    for name, model in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", model)])
        pipe.fit(Xt, yt)
        pred = pipe.predict(Xv)
        prob = pipe.predict_proba(Xv)[:, 1]
        pipes[name] = pipe
        cms[name] = confusion_matrix(yv, pred)
        rows[name] = {
            "Accuracy": accuracy_score(yv, pred),
            "Precision": precision_score(yv, pred, zero_division=0),
            "Recall": recall_score(yv, pred, zero_division=0),
            "F1-Score": f1_score(yv, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(yv, prob),
        }

    result = pd.DataFrame(rows).T.reset_index(names="Model")
    result = result.sort_values(["Recall", "F1-Score"], ascending=False).reset_index(drop=True)
    best = result.iloc[0]["Model"]

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cvrows = []
    for name, pipe in pipes.items():
        s = cross_validate(
            pipe, X, y, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )
        cvrows.append({
            "Model": name,
            "Accuracy": s["test_accuracy"].mean(),
            "Precision": s["test_precision"].mean(),
            "Recall": s["test_recall"].mean(),
            "F1-Score": s["test_f1"].mean(),
            "ROC-AUC": s["test_roc_auc"].mean(),
        })
    cvdf = pd.DataFrame(cvrows).sort_values(["Recall", "F1-Score"], ascending=False).reset_index(drop=True)

    return {
        "df": df, "X": X, "y": y, "pipes": pipes, "result": result,
        "cv": cvdf, "cms": cms, "best": best,
        "split": {"X_test": Xv, "y_test": yv},
    }


df = load_data()
ml = build_models(df)
results = ml["result"]
cv = ml["cv"]
auto_model = ml["best"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def fmt_pct(x):
    return f"{x:.1%}"


def active_model():
    choice = st.session_state.get("model_control", "Auto — Recall-first")
    return auto_model if choice == "Auto — Recall-first" else choice


def plotly_base(fig, height=250, margin=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#b7b7bc", size=11),
        height=height,
        margin=margin or dict(l=8, r=8, t=34, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hoverlabel=dict(bgcolor="#151518", bordercolor="#333337", font=dict(color="#f5f5f5")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#27272b", tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#1c1c20", zeroline=False, linecolor="#27272b", tickfont=dict(size=10))
    return fig


def metric_cards(items):
    cols = st.columns(len(items), gap="small")
    for col, (label, value, sub) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",
                unsafe_allow_html=True,
            )


def page_header(kicker, title, subtitle, right=None):
    left, rightcol = st.columns([4.7, 1.3])
    with left:
        st.markdown(f"<div class='eyebrow'>{kicker}</div><h1>{title}</h1><p class='lead'>{subtitle}</p>", unsafe_allow_html=True)
    if right:
        with rightcol:
            st.markdown(right, unsafe_allow_html=True)


def model_metrics(name):
    r = results[results.Model == name].iloc[0]
    return r


def risk_band(prob, threshold):
    if prob >= max(.75, threshold + .20):
        return "CRITICAL", "high"
    if prob >= threshold:
        return "HIGH", "high"
    if prob >= .30:
        return "WATCH", "watch"
    return "LOW", "low"


def demo_profiles(model):
    probs = model.predict_proba(ml["X"])[:, 1]
    hi = int(np.argmax(probs))
    lo = int(np.argmin(probs))
    return {
        "Highest-risk demo": ml["X"].iloc[hi].to_dict(),
        "Lowest-risk demo": ml["X"].iloc[lo].to_dict(),
    }, float(probs[hi]), float(probs[lo])


def render_model_badge(name):
    r = model_metrics(name)
    st.markdown(
        f"<div class='badge'><span class='dot'></span> Active model: {name} &nbsp;·&nbsp; Recall {fmt_pct(r.Recall)}</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — model control first
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div class='sidebar-brand'><div class='mark'>CHURN<span>IQ</span> ◈</div>"
    "<div class='caption'>Telecom intelligence</div></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='sidebar-section'>Workspace</div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Customer Analytics", "Model Lab", "Predictor", "Methodology"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<div class='sidebar-section'>Model control</div>", unsafe_allow_html=True)
model_options = ["Auto — Recall-first", "Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]
st.sidebar.selectbox(
    "Active model",
    model_options,
    key="model_control",
    label_visibility="collapsed",
    help="Auto chooses the model with the highest test Recall, using F1 as the secondary tie-breaker. You can override it for the dashboard and predictor.",
)
active = active_model()
active_row = model_metrics(active)

st.sidebar.markdown(
    f"<div class='card' style='padding:13px;margin-top:8px'>"
    f"<div class='metric-label'>ACTIVE</div><div style='font-family:Space Grotesk;font-weight:600;font-size:.95rem;margin-top:5px'>{active}</div>"
    f"<div class='metric-sub'>Recall {active_row.Recall:.1%} · F1 {active_row['F1-Score']:.1%}</div></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='sidebar-section'>Project</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div class='sidebar-note'><b style='color:#aaa'>IBM Telco sample</b><br>{len(df):,} customers · {df.shape[1]} raw columns<br><br>Primary criterion: <b style='color:#aaa'>Recall</b><br>Five required classifiers</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Machine learning case study · 20 marks</div>"
        "<div class='hero-title'>Know who may leave.<br><span>Understand the signal.</span></div>"
        "<div class='hero-sub'>A compact telecom churn intelligence workspace covering data exploration, five classifiers, cross-validation and an explainable customer-level predictor. Built for a clear academic demonstration — not just a form.</div></div>",
        unsafe_allow_html=True,
    )
    render_model_badge(active)
    st.markdown("<div class='section-head'><div><div class='kicker'>At a glance</div><div class='title'>Project telemetry</div></div></div>", unsafe_allow_html=True)
    churn = (df.Churn == "Yes").mean()
    metric_cards([
        ("Customers", f"{len(df):,}", "IBM sample records"),
        ("Observed churn", f"{churn:.1%}", "Target = Churn"),
        ("Algorithms", "05", "Required classifiers"),
        ("Active model", active, "Recall-first control"),
    ])

    st.markdown("<div class='section-head'><div><div class='kicker'>Signal overview</div><div class='title'>Where churn concentrates</div></div><div class='muted'>Compact view · hover for exact values</div></div>", unsafe_allow_html=True)
    a, b = st.columns(2, gap="small")
    with a:
        contract = pd.crosstab(df.Contract, df.Churn)
        fig = go.Figure()
        for label in ["No", "Yes"]:
            fig.add_trace(go.Bar(name="Stayed" if label == "No" else "Churned", x=contract.index, y=contract.get(label, 0), marker_color="#6e6e73" if label == "No" else "#d8d8dc"))
        fig.update_layout(barmode="group", title="Contract vs churn", xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(plotly_base(fig, 255, dict(l=5,r=5,t=40,b=8)), use_container_width=True, config={"displayModeBar": False})
    with b:
        rate = df.groupby("Contract", observed=True).Churn.apply(lambda s: (s == "Yes").mean()).reset_index(name="ChurnRate")
        fig = px.bar(rate, x="Contract", y="ChurnRate", text=rate.ChurnRate.map(lambda x: f"{x:.0%}"))
        fig.update_traces(marker_color="#d8d8dc", textposition="outside", cliponaxis=False)
        fig.update_yaxes(tickformat=".0%", range=[0, max(rate.ChurnRate.max() * 1.25, .55)])
        fig.update_layout(title="Observed churn rate by contract", xaxis_title=None, yaxis_title=None, showlegend=False)
        st.plotly_chart(plotly_base(fig, 255, dict(l=5,r=5,t=40,b=8)), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-head'><div><div class='kicker'>Workflow</div><div class='title'>From raw customer record to decision</div></div></div>", unsafe_allow_html=True)
    cols = st.columns(5, gap="small")
    steps = [
        ("01", "Explore", "Churn, contracts, tenure, charges and services."),
        ("02", "Prepare", "Numeric cleaning, imputation, encoding and scaling."),
        ("03", "Train", "Logistic, KNN, Tree, Forest and Boosting."),
        ("04", "Validate", "Recall, F1, ROC-AUC, confusion matrix and 5-fold CV."),
        ("05", "Predict", "Probability → threshold → risk band → explanation."),
    ]
    for c, (num, title, desc) in zip(cols, steps):
        with c:
            st.markdown(f"<div class='card'><div class='metric-label'>{num}</div><h3 style='margin:9px 0 5px'>{title}</h3><div class='muted'>{desc}</div></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOMER ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Customer Analytics":
    page_header("Data exploration", "Customer Analytics", "A tighter EDA view: filter the sample, inspect churn structure and then drill into service patterns.")

    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.1, .8], gap="small")
    with f1: contract_filter = st.multiselect("Contract", sorted(df.Contract.unique()), default=list(sorted(df.Contract.unique())))
    with f2: internet_filter = st.multiselect("Internet service", sorted(df.InternetService.unique()), default=list(sorted(df.InternetService.unique())))
    with f3: outcome_filter = st.multiselect("Outcome", ["No", "Yes"], default=["No", "Yes"])
    with f4:
        st.markdown("<div class='metric-label' style='margin-top:3px'>Rows shown</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:Space Grotesk;font-size:1.25rem;font-weight:600;margin-top:7px'>{len(df):,}</div>", unsafe_allow_html=True)

    view = df[
        df.Contract.isin(contract_filter) &
        df.InternetService.isin(internet_filter) &
        df.Churn.isin(outcome_filter)
    ].copy()
    vr = (view.Churn == "Yes").mean() if len(view) else 0
    metric_cards([
        ("Filtered customers", f"{len(view):,}", "Current slice"),
        ("Churn rate", f"{vr:.1%}", "Within filtered slice"),
        ("Median tenure", f"{view.tenure.median():.0f} mo" if len(view) else "—", "Customer lifetime"),
        ("Median monthly", f"${view.MonthlyCharges.median():.0f}" if len(view) else "—", "Monthly charges"),
    ])

    a, b = st.columns(2, gap="small")
    with a:
        rates = view.groupby("Contract", observed=True).Churn.apply(lambda s: (s == "Yes").mean()).reset_index(name="Rate") if len(view) else pd.DataFrame(columns=["Contract","Rate"])
        fig = px.bar(rates, x="Contract", y="Rate", text=rates.Rate.map(lambda x: f"{x:.0%}") if len(rates) else None)
        fig.update_traces(marker_color="#d8d8dc", textposition="outside", cliponaxis=False)
        fig.update_yaxes(tickformat=".0%", range=[0, max(.55, (rates.Rate.max() * 1.25 if len(rates) else .55))])
        fig.update_layout(title="Churn rate by contract", xaxis_title=None, yaxis_title=None)
        st.plotly_chart(plotly_base(fig, 255, dict(l=5,r=5,t=40,b=8)), use_container_width=True, config={"displayModeBar": False})
    with b:
        fig = px.histogram(view, x="tenure", color="Churn", nbins=24, barmode="overlay") if len(view) else go.Figure()
        if len(view):
            fig.update_traces(opacity=.72)
            fig.update_layout(title="Tenure distribution", xaxis_title="Months", yaxis_title="Customers")
        st.plotly_chart(plotly_base(fig, 255, dict(l=5,r=5,t=40,b=8)), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-head'><div><div class='kicker'>Three-dimensional view</div><div class='title'>Customer charge / tenure landscape</div></div><div class='muted'>Sampled for responsiveness · color = churn</div></div>", unsafe_allow_html=True)
    sample = view.sample(min(1800, len(view)), random_state=42) if len(view) else view
    if len(sample):
        fig3d = px.scatter_3d(sample, x="tenure", y="MonthlyCharges", z="TotalCharges", color="Churn", opacity=.62, hover_data=["Contract", "InternetService", "PaymentMethod"])
        fig3d.update_traces(marker=dict(size=3.5))
        fig3d.update_layout(scene=dict(xaxis_title="Tenure", yaxis_title="Monthly charges", zaxis_title="Total charges"), legend_title_text="Churn")
        st.plotly_chart(plotly_base(fig3d, 440, dict(l=0,r=0,t=5,b=0)), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-head'><div><div class='kicker'>Service inspection</div><div class='title'>Attribute-level churn rate</div></div></div>", unsafe_allow_html=True)
    attr = st.selectbox("Inspect attribute", ["InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "Contract", "PaperlessBilling"])
    table = pd.crosstab(view[attr], view.Churn, normalize="index") * 100 if len(view) else pd.DataFrame()
    if len(table):
        table = table.rename(columns={"No": "Stayed %", "Yes": "Churned %"}).round(1)
        st.dataframe(table, use_container_width=True, hide_index=False)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LAB
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Model Lab":
    active_badge = "<div class='badge'><span class='dot'></span>" + active + "</div>"
    page_header("Evaluation workspace", "Model Lab", "See all five classifiers side-by-side, inspect the active model, and use the 3D view to understand the metric trade-off.", right="<div style='text-align:right'>" + active_badge + "</div>")

    display = results.copy()
    display["Status"] = np.where(display.Model == auto_model, "AUTO", "")
    styled = display.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        styled[c] = styled[c].map(lambda x: f"{x:.1%}")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    m1, m2 = st.columns(2, gap="small")
    with m1:
        metric = st.selectbox("Metric chart", ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], index=2)
        p = results.sort_values(metric, ascending=True)
        fig = px.bar(p, x=metric, y="Model", orientation="h", text=p[metric].map(lambda x: f"{x:.1%}"))
        fig.update_traces(marker_color="#d8d8dc", textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, min(1.05, max(.75, p[metric].max() * 1.12))], tickformat=".0%")
        fig.update_layout(title=f"{metric} by model", xaxis_title=None, yaxis_title=None)
        st.plotly_chart(plotly_base(fig, 285, dict(l=4,r=34,t=42,b=5)), use_container_width=True, config={"displayModeBar": False})
    with m2:
        st.markdown("<div class='card' style='height:285px'><div class='metric-label'>Why recall first?</div><h3>False negatives matter</h3><div class='small'>The case study prioritises finding customers who actually churn. Recall measures the share of actual churners that the model identifies.</div><div class='divider'></div><div class='small'><b style='color:#ddd'>Auto selection:</b> highest test Recall, then F1 as the tie-breaker.</div><div class='small' style='margin-top:8px'><b style='color:#ddd'>Current:</b> {}</div></div>".format(auto_model), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        cm_model = st.selectbox("Confusion matrix model", list(results.Model), index=list(results.Model).index(active))
        cm = ml["cms"][cm_model]
        z = [[int(cm[0,0]), int(cm[0,1])], [int(cm[1,0]), int(cm[1,1])]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"], text=z, texttemplate="%{text}", textfont=dict(size=15, color="#f5f5f5"), colorscale=[[0,"#171719"],[1,"#e5e5e5"]], showscale=False))
        fig.update_layout(title=f"Confusion matrix · {cm_model}", yaxis_autorange="reversed")
        st.plotly_chart(plotly_base(fig, 290, dict(l=5,r=5,t=40,b=5)), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("<div class='section-head' style='margin-top:0'><div><div class='kicker'>3D metric landscape</div><div class='title'>Precision · Recall · F1</div></div></div>", unsafe_allow_html=True)
        fig3 = px.scatter_3d(results, x="Precision", y="Recall", z="F1-Score", text="Model", size="ROC-AUC", hover_data=["Accuracy"])
        fig3.update_traces(marker=dict(size=7), textposition="top center")
        fig3.update_layout(scene=dict(xaxis_title="Precision", yaxis_title="Recall", zaxis_title="F1"))
        st.plotly_chart(plotly_base(fig3, 290, dict(l=0,r=0,t=8,b=0)), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-head'><div><div class='kicker'>Stability check</div><div class='title'>Stratified 5-fold cross-validation</div></div></div>", unsafe_allow_html=True)
    cvshow = cv.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        cvshow[c] = cvshow[c].map(lambda x: f"{x:.1%}")
    st.dataframe(cvshow, use_container_width=True, hide_index=True)
    st.info(f"Auto model = {auto_model}. It is selected by highest hold-out Recall, with F1 used as the secondary tie-breaker. The sidebar model control lets you deliberately inspect or deploy any required classifier.")

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Predictor":
    page_header("Decision-support workspace", "Customer Predictor", "Build a customer profile, choose the active classifier, set the decision threshold, and inspect probability plus the signals behind the result.")

    selected_pipe = ml["pipes"][active]
    presets, hi_prob, lo_prob = demo_profiles(selected_pipe)

    top1, top2, top3 = st.columns([1.5, 1.1, 1.1], gap="small")
    with top1:
        mode = st.radio("Profile mode", ["Custom profile", "Highest-risk demo", "Lowest-risk demo"], horizontal=True)
    with top2:
        threshold = st.slider("High-risk threshold", 0.30, 0.70, 0.50, 0.01, help="Probability cutoff used to turn the model probability into High/Low. It does not change the underlying model probability.")
    with top3:
        r = model_metrics(active)
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Using</div><div class='metric-value' style='font-size:1.05rem'>{active}</div><div class='metric-sub'>Recall {r.Recall:.1%} · F1 {r['F1-Score']:.1%}</div></div>", unsafe_allow_html=True)

    preset = presets.get(mode)
    def gv(key, default):
        return preset.get(key, default) if preset is not None else default

    with st.form("predict_form", clear_on_submit=False):
        st.markdown("<div class='section-head' style='margin-top:8px'><div><div class='kicker'>Customer profile</div><div class='title'>Subscription & service attributes</div></div></div>", unsafe_allow_html=True)
        a, b, c = st.columns(3, gap="small")
        with a:
            gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(gv("gender", "Female")))
            senior = st.selectbox("Senior citizen", [0, 1], index=int(gv("SeniorCitizen", 0)), format_func=lambda z: "Yes" if z else "No")
            partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(gv("Partner", "No")))
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(gv("Dependents", "No")))
            tenure = st.slider("Tenure (months)", 0, 72, int(gv("tenure", 12)))
            phone = st.selectbox("Phone service", ["Yes", "No"], index=["Yes", "No"].index(gv("PhoneService", "Yes")))
            lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"], index=["Yes", "No", "No phone service"].index(gv("MultipleLines", "No")))
        with b:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(gv("InternetService", "Fiber optic")))
            security = st.selectbox("Online security", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(gv("OnlineSecurity", "No")))
            backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(gv("OnlineBackup", "No")))
            protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(gv("DeviceProtection", "No")))
            tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(gv("TechSupport", "No")))
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(gv("StreamingTV", "No")))
            movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(gv("StreamingMovies", "No")))
        with c:
            contracts = ["Month-to-month", "One year", "Two year"]
            payments = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            contract = st.selectbox("Contract", contracts, index=contracts.index(gv("Contract", "Month-to-month")))
            paperless = st.selectbox("Paperless billing", ["Yes", "No"], index=["Yes", "No"].index(gv("PaperlessBilling", "Yes")))
            payment = st.selectbox("Payment method", payments, index=payments.index(gv("PaymentMethod", "Electronic check")))
            monthly = st.number_input("Monthly charges", 18.0, 150.0, float(gv("MonthlyCharges", 70.0)), .5)
            raw_total = gv("TotalCharges", monthly * tenure)
            default_total = float(raw_total) if pd.notna(raw_total) else float(monthly * tenure)
            default_total = min(10000.0, max(0.0, default_total))
            total = st.number_input("Total charges", 0.0, 10000.0, default_total, 10.0)
            st.markdown("<div class='muted' style='margin-top:10px'>Tip: demo modes load real records from the dataset, so you can reproduce clear High and Low examples without guessing a combination.</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("◈  ANALYSE CUSTOMER", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": tech, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        prob = float(selected_pipe.predict_proba(row)[0, 1])
        band, cls = risk_band(prob, threshold)
        risk = prob >= threshold

        st.markdown("<div class='section-head'><div><div class='kicker'>Prediction output</div><div class='title'>Decision layer</div></div></div>", unsafe_allow_html=True)
        left, right = st.columns([1.35, 1], gap="small")
        with left:
            st.markdown(
                f"<div class='risk-panel {cls}'><div class='metric-label'>Estimated churn risk · {band}</div>"
                f"<div class='risk-word'>{'HIGH' if risk else 'LOW'}</div>"
                f"<div class='risk-caption'>Model probability <b style='color:#ddd'>{prob:.1%}</b> · decision threshold <b style='color:#ddd'>{threshold:.0%}</b><br>Model: {active}</div></div>",
                unsafe_allow_html=True,
            )
        with right:
            metric_cards([
                ("Probability", f"{prob:.1%}", "Model output"),
                ("Threshold", f"{threshold:.0%}", "Decision rule"),
                ("Tenure", f"{tenure} mo", "Customer snapshot"),
            ])

        st.markdown("<div class='section-head'><div><div class='kicker'>Interpretation</div><div class='title'>Why the profile looks this way</div></div></div>", unsafe_allow_html=True)
        reasons = []
        if contract == "Month-to-month": reasons.append("Month-to-month contract is associated with higher churn in this sample.")
        if contract == "Two year": reasons.append("Two-year contract is associated with lower churn in this sample.")
        if tenure <= 12: reasons.append("Short tenure is associated with higher churn in this sample.")
        if tenure >= 48: reasons.append("Long tenure is associated with lower churn in this sample.")
        if monthly >= 80: reasons.append("Higher monthly charges are associated with higher churn in this sample.")
        if monthly <= 50: reasons.append("Lower monthly charges are associated with lower churn in this sample.")
        if payment == "Electronic check": reasons.append("Electronic-check payment is associated with higher churn in this sample.")
        if internet == "Fiber optic": reasons.append("Fiber-optic service is a meaningful categorical pattern in the fitted model/data.")
        if tech == "Yes": reasons.append("Tech support availability changes the service profile considered by the model.")
        if not reasons: reasons.append("The probability comes from the combined pattern of all supplied customer attributes.")

        e1, e2 = st.columns([1.15, .85], gap="small")
        with e1:
            st.markdown("<div class='explain'><div class='metric-label'>Signal explanation</div>" + "".join([f"<div class='signal'><span class='bullet'></span><span>{x}</span></div>" for x in reasons[:6]]) + "</div>", unsafe_allow_html=True)
        with e2:
            st.markdown("<div class='card'><div class='metric-label'>Threshold mechanics</div><h3>Probability ≠ label</h3><div class='small'>The model first outputs a probability. The threshold then converts that probability into High or Low.</div><div class='divider'></div><div class='small'><b style='color:#ddd'>{prob:.1%}</b> ≥ <b style='color:#ddd'>{threshold:.0%}</b> → <b style='color:#ddd'>HIGH</b></div><div class='small' style='margin-top:6px'>Raise the threshold to make High classification more selective; lower it to flag more customers.</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-head'><div><div class='kicker'>Customer record</div><div class='title'>Snapshot used by the predictor</div></div></div>", unsafe_allow_html=True)
        snap = pd.DataFrame({
            "Attribute": ["Contract", "Tenure", "Internet", "Payment", "Monthly charges", "Total charges", "Senior citizen", "Tech support"],
            "Value": [contract, f"{tenure} months", internet, payment, f"${monthly:.2f}", f"${total:.2f}", "Yes" if senior else "No", tech],
        })
        st.dataframe(snap, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-head'><div><div class='kicker'>Responsible interpretation</div><div class='title'>What this result does — and does not — mean</div></div></div>", unsafe_allow_html=True)
        st.warning("This is a predictive classification for the educational sample dataset. The explanation describes model/data associations, not causal effects. In a production system, threshold selection should be tied to business costs, probability calibration, fairness review and monitoring for data drift.")

# ──────────────────────────────────────────────────────────────────────────────
# METHODOLOGY
# ──────────────────────────────────────────────────────────────────────────────
else:
    page_header("Project documentation", "Methodology", "A concise technical reference for the viva: data, preprocessing, algorithms, evaluation and responsible interpretation.")
    tabs = st.tabs(["Dataset", "Preprocessing", "Algorithms", "Metrics", "Limitations"])
    with tabs[0]:
        metric_cards([
            ("Rows", f"{len(df):,}", "IBM Telco sample"),
            ("Raw columns", str(df.shape[1]), "Before dropping ID"),
            ("Target", "Churn", "Binary classification"),
            ("Blank TotalCharges", str(df.TotalCharges.isna().sum()), "Converted from text"),
        ])
        st.markdown("### Dataset structure")
        st.markdown("<div class='card'><div class='small'>The dataset contains customer demographics, tenure, contract information, services, payment method, monthly charges, total charges and the Churn target. <b>customerID</b> is an identifier and is excluded from modelling.</div></div>", unsafe_allow_html=True)
    with tabs[1]:
        st.code("""TotalCharges → numeric\n\nNumeric features\n  → median imputation\n  → StandardScaler\n\nCategorical features\n  → most-frequent imputation\n  → OneHotEncoder(handle_unknown='ignore')\n\nEverything is wrapped in ColumnTransformer + Pipeline.""")
        st.caption("Keeping preprocessing inside the pipeline prevents leakage between training and validation folds.")
    with tabs[2]:
        algos = pd.DataFrame({
            "Algorithm": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Gradient Boosting"],
            "Role": ["Linear baseline", "Distance-based", "Rule-based tree", "Bagged ensemble", "Sequential boosting"],
            "Primary idea": ["Weighted linear decision boundary", "Nearest training examples", "Recursive feature splits", "Many randomized trees", "Trees correcting previous errors"],
        })
        st.dataframe(algos, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.markdown("### Evaluation logic")
        st.markdown("<div class='card'><div class='small'><b style='color:#eee'>Accuracy</b> — overall correct classifications.<br><br><b style='color:#eee'>Precision</b> — how many predicted churners actually churned.<br><br><b style='color:#eee'>Recall</b> — how many actual churners were found.<br><br><b style='color:#eee'>F1</b> — harmonic mean of precision and recall.<br><br><b style='color:#eee'>ROC-AUC</b> — ranking/discrimination across thresholds.<br><br><b style='color:#eee'>Confusion matrix</b> — TP, TN, FP and FN counts.<br><br><b style='color:#eee'>5-fold CV</b> — stability across stratified folds.</div></div>", unsafe_allow_html=True)
        st.success(f"Primary selection rule: highest hold-out Recall, then F1. Current automatic model: {auto_model}.")
    with tabs[4]:
        st.markdown("### Boundaries of the project")
        for item in [
            "IBM Telco data is a sample dataset for a fictional telecom company; it is not live production data.",
            "The analysis is predictive and observational; associations should not be interpreted as causal effects.",
            "Production deployment would require calibration, threshold tuning against business costs, privacy controls, fairness checks and drift monitoring.",
            "A model trained on historical sample data may not transfer directly to another telecom population without validation.",
        ]:
            st.markdown(f"- {item}")

# Footer
st.markdown("<div style='height:28px'></div><div style='border-top:1px solid #171719;padding-top:12px;color:#55555a;font-size:.66rem;text-align:center;letter-spacing:.04em'>CHURNIQ · TELECOM CUSTOMER CHURN · ACADEMIC ML DEMONSTRATION</div>", unsafe_allow_html=True)
