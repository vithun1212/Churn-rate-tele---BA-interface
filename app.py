import json
import math
import re

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn & Revenue-at-Risk", page_icon="📉", layout="centered")


@st.cache_resource
def load():
    return (joblib.load("linear_model.sav"), joblib.load("logistic_model.sav"),
            joblib.load("model_meta.sav"))


lin, logit, meta = load()
opt, kb = meta["cat_options"], meta["kb"]
THR = meta["log_metrics"]["threshold"]
LR = kb["logistic_regression"]

# ---------------------------------------------------------------- shared model logic
ADDONS = {"Online security": "OnlineSecurity", "Tech support": "TechSupport", "Online backup": "OnlineBackup",
          "Device protection": "DeviceProtection", "Streaming TV": "StreamingTV", "Streaming movies": "StreamingMovies",
          "Multiple phone lines": "MultipleLines"}
WIDGET_DEFAULTS = {"w_gender": "Female", "w_senior": False, "w_partner": False, "w_dependents": False,
                   "w_tenure": 12, "w_contract": "Month-to-month", "w_paperless": True,
                   "w_payment": "Electronic check", "w_phone": True, "w_internet": "DSL", "w_addons": [],
                   "w_use_actual": False, "w_actual": 70.0}
for k, v in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(k, v)


def score(c):
    """c: dict with all profile + service fields. Returns dict of model outputs."""
    c = dict(c)
    if c["PhoneService"] == "No":
        c["MultipleLines"] = "No"
    if c["InternetService"] == "No":
        for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            c[s] = "No"
    est = max(0.0, float(lin.predict(pd.DataFrame([{k: c[k] for k in meta["lin_cats"]}]))[0]))
    monthly = c.get("MonthlyOverride") or est
    row = pd.DataFrame([{"SeniorCitizen": 1 if c["SeniorCitizen"] == "Yes" else 0, "tenure": c["tenure"],
                         "MonthlyCharges": monthly, **{k: c[k] for k in meta["log_cats"]}}])
    p = float(logit.predict_proba(row)[0, 1])
    return {"customer": c, "estimated_monthly_charge": round(est, 2), "monthly_charge_used": round(monthly, 2),
            "churn_probability": round(p, 4), "high_risk": p >= THR,
            "monthly_revenue_at_risk": round(p * monthly, 2), "annual_revenue_at_risk": round(p * monthly * 12, 2)}


def current_form():
    ss = st.session_state
    yn = lambda x: "Yes" if x else "No"
    chosen = set(ss["w_addons"] or [])
    form = {"gender": ss["w_gender"] or "Female", "SeniorCitizen": yn(ss["w_senior"]), "Partner": yn(ss["w_partner"]),
            "Dependents": yn(ss["w_dependents"]), "tenure": ss["w_tenure"],
            "Contract": ss["w_contract"] or "Month-to-month", "PaperlessBilling": yn(ss["w_paperless"]),
            "PaymentMethod": ss["w_payment"] or "Electronic check", "PhoneService": yn(ss["w_phone"]),
            "InternetService": ss["w_internet"] or "DSL",
            "MonthlyOverride": ss["w_actual"] if ss["w_use_actual"] else None}
    form.update({col: yn(label in chosen) for label, col in ADDONS.items()})
    return form


def suggested_actions(c, r):
    """Plain-language retention ideas, each tied to a model odds ratio or a segment churn rate."""
    o, seg, acts = LR["odds_ratios"], kb["segments_churn_rate"], []
    if c["Contract"] == "Month-to-month":
        acts.append(f"**Offer a 1- or 2-year contract.** Longer contracts are linked to much lower churn "
                    f"(odds ratios {o['Contract_One year']:.2f} and {o['Contract_Two year']:.2f}).")
    if c["tenure"] <= 12:
        acts.append(f"**Early-life check-in call.** {seg['TenureBand']['0-12 months']['churn_rate_pct']}% of first-year customers leave.")
    if c["PaymentMethod"] == "Electronic check":
        acts.append(f"**Move to auto-pay.** E-check payers churn at {seg['PaymentMethod']['Electronic check']['churn_rate_pct']}% "
                    f"vs {seg['PaymentMethod']['Bank transfer (automatic)']['churn_rate_pct']}% for bank auto-pay.")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        acts.append(f"**Free trial of online security** (odds ratio {o['OnlineSecurity_Yes']:.2f}).")
    if c["InternetService"] != "No" and c["TechSupport"] == "No":
        acts.append(f"**Offer tech support** (odds ratio {o['TechSupport_Yes']:.2f}).")
    if c["InternetService"] == "Fiber optic":
        acts.append(f"**Check fiber service quality.** Fiber customers churn more even at the same price "
                    f"(odds ratio {o['InternetService_Fiber optic']:.2f}).")
    return acts


# ---------------------------------------------------------------- chat: parse a customer from text
def parse_customer(text):
    t = " " + text.lower().replace("fibre", "fiber") + " "
    f = {}
    # contract first, then remove it so "2 year contract" is not read as tenure
    for pat, val in [(r"month[\s-]*to[\s-]*month|monthly contract|no contract", "Month-to-month"),
                     (r"(two|2)[\s-]*years?[\s-]*(contract|plan|deal)", "Two year"),
                     (r"(one|1)[\s-]*years?[\s-]*(contract|plan|deal)|annual contract", "One year")]:
        m = re.search(pat, t)
        if m:
            f["Contract"] = val
            t = t.replace(m.group(0), " ")
            break
    m = re.search(r"(\d+)\s*(?:months?|mos?)\b", t)
    if m:
        f["tenure"] = min(72, int(m.group(1)))
    else:
        m = re.search(r"(\d+(?:\.\d+)?)[\s-]*(?:years?|yrs?)\b", t)
        if m:
            f["tenure"] = min(72, round(float(m.group(1)) * 12))
        elif re.search(r"\bnew (customer|user|joiner)|just joined", t):
            f["tenure"] = 1
    if "fiber" in t:
        f["InternetService"] = "Fiber optic"
    elif re.search(r"\bdsl\b", t):
        f["InternetService"] = "DSL"
    elif re.search(r"no internet|without internet", t):
        f["InternetService"] = "No"
    for pat, val in [(r"electronic check|e-?check", "Electronic check"), (r"mailed check|cheque", "Mailed check"),
                     (r"credit card", "Credit card (automatic)"), (r"bank transfer|auto[\s-]?pay", "Bank transfer (automatic)")]:
        if re.search(pat, t):
            f["PaymentMethod"] = val
            break
    if re.search(r"paper bill|no paperless|not paperless", t):
        f["PaperlessBilling"] = "No"
    elif "paperless" in t:
        f["PaperlessBilling"] = "Yes"
    if re.search(r"not (a )?senior|young", t):
        f["SeniorCitizen"] = "No"
    elif re.search(r"senior|elderly|retired", t):
        f["SeniorCitizen"] = "Yes"
    if re.search(r"\bfemale\b|\bwoman\b|\blady\b", t):
        f["gender"] = "Female"
    elif re.search(r"\bmale\b|\bman\b", t):
        f["gender"] = "Male"
    for key, words in [("Dependents", r"dependents?|kids|children"), ("Partner", r"partner|married|spouse")]:
        if re.search(r"(no|without|not)\s+(a\s+)?(" + words + ")", t) or (key == "Partner" and re.search(r"\bsingle\b", t)):
            f[key] = "No"
        elif re.search(words, t):
            f[key] = "Yes"
    addons = [("OnlineSecurity", r"(online )?security"), ("TechSupport", r"tech(nical)? support|support"),
              ("OnlineBackup", r"(online )?backup"), ("DeviceProtection", r"device protection"),
              ("StreamingTV", r"streaming tv|\btv\b"), ("StreamingMovies", r"streaming movies|movies"),
              ("MultipleLines", r"multiple lines"), ("PhoneService", r"phone( service)?")]
    for key, words in addons:
        if re.search(r"(no|without|not|lacks?|doesn'?t have)\s+(any\s+)?(" + words + ")", t):
            f[key] = "No"
        elif re.search(words, t):
            f[key] = "Yes"
    if re.search(r"(no|without) (any )?add[\s-]?ons", t):
        for key, _ in addons[:6]:
            f[key] = "No"
    if re.search(r"\bstreaming\b", t) and not re.search(r"(no|without)\s+streaming", t):
        f.setdefault("StreamingTV", "Yes")
        f.setdefault("StreamingMovies", "Yes")
    m = re.search(r"(?:pays?|paying|bill(?:ed)?|charges?d?|monthly charge(?: of)?)\D{0,12}(\d{2,3}(?:\.\d+)?)(?![\d.]|\s*(?:months?|mos?|years?))", t)
    if m:
        f["MonthlyOverride"] = float(m.group(1))
    return f


PRED_INTENT = re.compile(r"predict|what if|what-if|chance|probab|likely|risk (of|for|score)|will (he|she|they|this|the customer)|"
                         r"\bscore\b|\ba customer\b|customer (who|with|on|having|paying)|someone|this customer", re.I)


# ---------------------------------------------------------------- chat: offline (rule-based) answers
def pct(x):
    return f"{x * 100:.1f}%"


def fmt_prediction(r, parsed):
    c = r["customer"]
    lines = [f"**Churn probability: {pct(r['churn_probability'])}** → "
             f"{'🔴 High risk (≥ ' + pct(THR) + ')' if r['high_risk'] else '🟢 Low risk (< ' + pct(THR) + ')'}",
             f"- Monthly charge used: **{r['monthly_charge_used']}** "
             f"({'given in your question' if 'MonthlyOverride' in parsed else 'Linear Regression estimate for the bundle'})",
             f"- Revenue at risk: **{r['monthly_revenue_at_risk']} / month**, **{r['annual_revenue_at_risk']} / year**",
             "", "| Field | Value | Source |", "|---|---|---|"]
    for k in ["tenure", "Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "OnlineSecurity",
              "TechSupport", "StreamingTV", "StreamingMovies", "PhoneService", "SeniorCitizen", "Partner", "Dependents", "gender"]:
        lines.append(f"| {k} | {c[k]} | {'your question' if k in parsed else 'form'} |")
    lines.append("\n_Fields you didn't mention are taken from the current 'Check a customer' tab settings._")
    return "\n".join(lines)


SEG_KEYS = [("Contract", r"contract|month[\s-]*to[\s-]*month|one year|two year|annual"),
            ("InternetService", r"internet|fiber|dsl"), ("PaymentMethod", r"payment|check|cheque|credit card|bank transfer|pay"),
            ("PaperlessBilling", r"paperless|billing"), ("SeniorCitizen", r"senior|elderly|age"),
            ("TenureBand", r"tenure|new customer|loyal|months|years"), ("OnlineSecurity", r"security"),
            ("TechSupport", r"support"), ("StreamingTV", r"\btv\b|streaming"), ("StreamingMovies", r"movies|streaming"),
            ("OnlineBackup", r"backup"), ("DeviceProtection", r"device"), ("PhoneService", r"phone"),
            ("MultipleLines", r"multiple lines"), ("gender", r"gender|male|female|women|men"),
            ("Partner", r"partner|married"), ("Dependents", r"dependents|children|kids")]


def seg_table(col):
    rows = kb["segments_churn_rate"][col]
    out = [f"**Churn rate by {col}**", "", "| Group | Churn rate | Customers |", "|---|---|---|"]
    for g, v in sorted(rows.items(), key=lambda x: -x[1]["churn_rate_pct"]):
        out.append(f"| {g} | {v['churn_rate_pct']}% | {v['customers']:,} |")
    return "\n".join(out)


def offline_answer(q):
    t = q.lower()
    d, lr, li = kb["dataset"], LR, kb["linear_regression"]
    if re.search(r"accura|auc|roc|recall|precision|perform|metric|how good|confusion", t):
        cm = lr["confusion_matrix_at_threshold"]
        return (f"**Logistic Regression (churn), test set of 1,409:** ROC-AUC **{lr['roc_auc']}** "
                f"(5-fold CV {lr['cv_auc_mean']} ± {lr['cv_auc_sd']}), at the {pct(THR)} threshold: accuracy {lr['accuracy']}, "
                f"precision {lr['precision']}, recall {lr['recall']}.\n\nConfusion matrix: {cm['caught_churner']} churners caught, "
                f"{cm['missed_churner']} missed, {cm['false_alarm']} false alarms, {cm['true_stay']} correct 'stay'.\n\n"
                f"**Linear Regression (monthly charge):** R² **{li['r2']}**, MAE **{li['mae']}**.")
    if re.search(r"threshold|cut[\s-]?off|0\.3|30 ?%", t):
        rows = ["| Threshold | Precision | Recall | Accuracy | Flagged |", "|---|---|---|---|---|"]
        rows += [f"| {r['threshold']} | {r['precision']} | {r['recall']} | {r['accuracy']} | {r['flagged']} |" for r in kb["threshold_table"]]
        return (f"Customers are flagged when churn probability ≥ **{THR}**. A missed churner loses full revenue while a false alarm "
                "only costs an offer, so recall is prioritised over accuracy.\n\n" + "\n".join(rows))
    if re.search(r"calibrat|decile|lift", t):
        rows = ["| Risk decile | Mean predicted | Actual churn |", "|---|---|---|"]
        rows += [f"| {r['risk_decile']} | {r['mean_predicted_pct']}% | {r['actual_churn_pct']}% |" for r in kb["calibration_by_decile"]]
        return "Predicted and actual churn by risk decile (1 = highest risk):\n\n" + "\n".join(rows)
    if re.search(r"revenue|at risk|money|loss|target|priorit", t):
        r = kb["revenue_at_risk_test_set"]
        return (f"On the test set ({r['test_customers']:,} customers): monthly revenue {r['total_monthly_revenue']:,.0f}, "
                f"actually lost to churn {r['actual_monthly_revenue_lost']:,.0f}, model's predicted revenue at risk "
                f"{r['predicted_revenue_at_risk']:,.0f}.\n\nTargeting the top 20% by revenue at risk catches "
                f"**{r['top20pct_by_revenue_at_risk_churners_caught_pct']}%** of churners and "
                f"**{r['top20pct_share_of_lost_revenue_pct']}%** of lost revenue.")
    if re.search(r"price|\bcosts?\b|coefficient|linear|how much|charge for|add[\s-]?on", t) and "odds" not in t:
        rows = [f"Base plan (DSL, no phone, no add-ons): **{li['intercept_base_plan']}**", "",
                "| Service | Adds to monthly charge |", "|---|---|"]
        rows += [f"| {k} | {v:+.2f} |" for k, v in li["coefficients"].items()]
        return "**Linear Regression price list** (R² " + str(li["r2"]) + "):\n\n" + "\n".join(rows)
    if re.search(r"odds|driver|why .*churn|factor|reason|cause|influence|important|reduce churn|retain|retention", t):
        o = lr["odds_ratios"]
        rows = ["| Variable | Odds ratio | Effect |", "|---|---|---|"]
        rows += [f"| {k} | {v:.2f} | {'raises' if v > 1 else 'lowers'} odds by {abs(v - 1) * 100:.0f}% |"
                 for k, v in sorted(o.items(), key=lambda x: -abs(math.log(x[1])))]
        return ("**Churn drivers (Logistic Regression odds ratios)**, baselines: "
                + ", ".join(f"{k} = {v}" for k, v in lr["baseline_categories"].items() if k in ["Contract", "InternetService", "PaymentMethod"])
                + ".\n\n" + "\n".join(rows) + "\n\n_" + lr["odds_ratio_note"] + "_")
    if re.search(r"limitation|weakness|drawback|improve|next step", t):
        return "**Limitations:**\n" + "\n".join("- " + x for x in kb["limitations"])
    if re.search(r"how does|approach|method|why two|logic|explain the (app|model)|design|why (drop|exclude|remove)", t):
        return kb["about"] + "\n\n**Design choices:**\n" + "\n".join("- " + x for x in kb["design_choices"])
    hits = [c for c, pat in SEG_KEYS if re.search(pat, t)]
    if hits or re.search(r"churn rate|how many|overall|dataset", t):
        head = (f"Overall: **{d['churn_rate_pct']}%** churn ({d['churners']:,} of {d['customers']:,}). "
                f"Churners average {d['avg_tenure_months']['churned']} months tenure vs {d['avg_tenure_months']['stayed']} for stayers, "
                f"and pay {d['avg_monthly_charge']['churned']} vs {d['avg_monthly_charge']['stayed']} a month.")
        seen, parts = set(), [head]
        for c in hits[:3]:
            if c not in seen:
                seen.add(c)
                parts.append(seg_table(c))
        return "\n\n".join(parts)
    return None


HELP = ("I can answer from the trained models and the data. Try:\n"
        "- *Predict: new customer, fiber, month-to-month, electronic check, no tech support*\n"
        "- *What if a 3-year customer on a two year contract with DSL pays 60?*\n"
        "- *Churn rate by contract* · *Which factors drive churn?* · *What does each service cost?*\n"
        "- *How accurate is the model?* · *Why a 30% threshold?* · *Revenue at risk* · *Limitations*")


# ---------------------------------------------------------------- chat: optional Gemini AI mode
def gemini_key():
    k = st.session_state.get("user_key")
    if k:
        return k
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def gemini_models():
    try:
        m = st.secrets.get("GEMINI_MODEL")
    except Exception:
        m = None
    return [x for x in [st.session_state.get("ok_model"), m, "gemini-flash-lite-latest",
                        "gemini-2.5-flash-lite", "gemini-2.5-flash"] if x]


SYSTEM = ("You are the analytics assistant inside a telecom customer-churn Streamlit app built for an MBA Business Analytics course. "
          "Answer ONLY from the KNOWLEDGE BASE JSON and any MODEL PREDICTION given below. Never invent or estimate numbers; "
          "if something is not covered, say it is not available in this model's outputs. Charges have no stated currency: call them 'units'. "
          "Describe odds ratios as associations, not causal effects. Numeric odds ratios are per 1 standard deviation. "
          "If a MODEL PREDICTION is provided, explain it using the odds ratios (which inputs push risk up or down) and suggest a retention action; "
          "do not change its numbers. Be concise, use short markdown tables where helpful.\n\nKNOWLEDGE BASE:\n")


def ask_gemini(question, history, prediction):
    key = gemini_key()
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None
    client = genai.Client(api_key=key)
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
    prompt = (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "") + (
        f"MODEL PREDICTION (computed by the app, exact):\n{json.dumps(prediction)}\n\n" if prediction else "") + f"USER: {question}"
    for model in gemini_models():
        try:
            r = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM + json.dumps(kb), temperature=0.2,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
            if r.text:
                st.session_state["ok_model"] = model
                return r.text
        except Exception as e:
            st.session_state["ai_error"] = f"{model}: {str(e)[:160]}"
    return None


def answer(q):
    parsed = parse_customer(q)
    pred = None
    parts = []
    if parsed and (PRED_INTENT.search(q) or q.lower().startswith("predict")):
        base = current_form()
        base["MonthlyOverride"] = None          # form's actual bill belongs to the form's bundle, not this one
        base.update(parsed)
        pred = score(base)
        parts.append(fmt_prediction(pred, parsed))
    ai = ask_gemini(q, st.session_state.messages[:-1], pred)
    if ai:
        parts.append(ai)
        return "\n\n---\n\n".join(parts) if pred else ai, "ai"
    if not pred:
        off = offline_answer(q)
        parts.append(off if off else "I couldn't match that to the model's outputs.\n\n" + HELP)
    return "\n\n".join(parts), "offline"


# ================================================================= UI
st.markdown("""
<style>
div[data-testid="stSegmentedControl"] button {min-width: 90px;}
.small-muted {color: #6b7280; font-size: 0.85rem;}
</style>""", unsafe_allow_html=True)

st.title("📉 Churn & Revenue-at-Risk Checker")
with st.container(border=True):
    st.markdown("**What this tool does:** it estimates how likely a customer is to leave and how much monthly revenue "
                "that puts at risk, so retention effort goes first to the customers who matter most.")
    i1, i2, i3 = st.columns(3)
    i1.markdown("**① Describe the customer**  \n<span class='small-muted'>Click the options below</span>", unsafe_allow_html=True)
    i2.markdown("**② Price the plan**  \n<span class='small-muted'>Linear Regression estimates the monthly bill</span>", unsafe_allow_html=True)
    i3.markdown("**③ Score the risk**  \n<span class='small-muted'>Logistic Regression gives the chance of leaving</span>", unsafe_allow_html=True)
    with st.expander("How reliable is it?"):
        m = meta["log_metrics"]
        st.markdown(
            f"- Built from **{kb['dataset']['customers']:,} telecom customers**, of whom {kb['dataset']['churn_rate_pct']}% left "
            "(IBM's public sample dataset; charges have no stated currency).\n"
            f"- Of every 4 customers who actually leave, it flags about **{round(m['recall'] * 4)}**. "
            f"About **{round(m['precision'] * 100)}%** of flagged customers really do leave.\n"
            f"- It ranks a leaver above a stayer **{m['roc_auc']:.0%}** of the time (ROC-AUC).\n"
            f"- The monthly-bill estimate is typically within **{meta['lin_metrics']['mae']}** of the actual bill.\n"
            f"- A customer is marked **high risk** at a {THR:.0%} or higher chance of leaving.")

with st.sidebar:
    st.subheader("💬 Chat mode")
    st.text_input("Gemini API key (optional)", type="password", key="user_key",
                  help="Free key from aistudio.google.com. Kept only for this browser session. "
                       "Or add GEMINI_API_KEY in Streamlit Cloud → Settings → Secrets.")
    if gemini_key():
        st.success("AI mode: Gemini answers from the model's knowledge base")
    else:
        st.info("Offline mode: rule-based answers from the model's outputs (no key needed)")
    if st.session_state.get("ai_error"):
        st.caption("Last AI error: " + st.session_state["ai_error"])
    if st.button("Clear chat"):
        st.session_state.messages = []

tab_pred, tab_chat = st.tabs(["🔮 Check a customer", "💬 Ask the model"])

CONTRACT_LBL = {"Month-to-month": "Month-to-month", "One year": "1 year", "Two year": "2 years"}
PAY_LBL = {"Electronic check": "E-check", "Mailed check": "Mailed cheque",
           "Bank transfer (automatic)": "Bank auto-pay", "Credit card (automatic)": "Card auto-pay"}
NET_LBL = {"DSL": "DSL", "Fiber optic": "Fiber", "No": "None"}

with tab_pred:
    st.markdown("#### 1. Customer")
    st.slider("Months with us", 0, 72, key="w_tenure")
    a1, a2 = st.columns([1.4, 1])
    a1.segmented_control("Contract", list(CONTRACT_LBL), format_func=CONTRACT_LBL.get, key="w_contract", required=True)
    a2.segmented_control("Gender", ["Female", "Male"], key="w_gender", required=True)
    st.segmented_control("Pays by", list(PAY_LBL), format_func=PAY_LBL.get, key="w_payment", required=True, wrap=True)
    t1, t2, t3, t4 = st.columns(4)
    t1.toggle("Senior citizen", key="w_senior")
    t2.toggle("Has a partner", key="w_partner")
    t3.toggle("Has dependents", key="w_dependents")
    t4.toggle("Paperless billing", key="w_paperless")

    st.markdown("#### 2. Services")
    b1, b2 = st.columns([1.4, 1])
    b1.segmented_control("Internet", list(NET_LBL), format_func=NET_LBL.get, key="w_internet", required=True)
    with b2:
        st.write("")
        st.toggle("Phone line", key="w_phone")
    st.pills("Add-ons (click all that apply)", list(ADDONS), selection_mode="multi", key="w_addons", wrap=True)
    if st.session_state["w_internet"] == "No":
        st.caption("Without internet, only 'Multiple phone lines' applies; other add-ons are ignored.")
    if not st.session_state["w_phone"]:
        st.caption("Without a phone line, 'Multiple phone lines' is ignored.")

    est = score({**current_form(), "MonthlyOverride": None})["estimated_monthly_charge"]
    st.markdown("#### 3. Monthly bill")
    c1, c2 = st.columns([1.2, 1])
    c1.markdown(f"Estimated from the services: **{est:.2f}** per month")
    c2.toggle("Use the actual bill instead", key="w_use_actual")
    if st.session_state["w_use_actual"]:
        c2.number_input("Actual monthly bill", 0.0, 200.0, step=0.5, key="w_actual")

    form = current_form()
    r = score(form)
    st.markdown("#### Result")
    with st.container(border=True):
        k1, k2, k3 = st.columns(3)
        k1.metric("Chance of leaving", f"{r['churn_probability']:.0%}")
        k2.metric("Revenue at risk / month", f"{r['monthly_revenue_at_risk']:.2f}")
        k3.metric("Revenue at risk / year", f"{r['annual_revenue_at_risk']:.2f}")
        st.progress(r["churn_probability"])
        if r["high_risk"]:
            st.error(f"🔴 **High risk.** Prioritise this customer for a retention offer.")
        else:
            st.success(f"🟢 **Low risk.** No immediate action needed.")
        acts = suggested_actions(r["customer"], r)
        if acts and r["high_risk"]:
            st.markdown("**Suggested actions** (based on what the model links to lower churn):")
            for x in acts[:4]:
                st.markdown("- " + x)
        elif acts:
            with st.expander("Ways to lower risk further"):
                for x in acts:
                    st.markdown("- " + x)
        st.caption("Revenue at risk = chance of leaving × monthly bill. Suggestions reflect patterns in the data, not proven cause and effect.")

    with st.expander("What drives churn overall? (odds ratios)"):
        st.write("Odds ratio above 1 raises the chance of leaving; below 1 lowers it.")
        o1, o2 = st.columns(2)
        o1.write("**Raises risk**")
        o1.dataframe(pd.Series(meta["top_risk"], name="Odds ratio"))
        o2.write("**Lowers risk**")
        o2.dataframe(pd.Series(meta["top_protect"], name="Odds ratio"))

with tab_chat:
    st.caption("Ask about the data, the models, or describe a customer to score. "
               "Predictions always come from the trained models; unmentioned fields use the 'Check a customer' tab settings.")
    st.session_state.setdefault("messages", [])
    examples = ["Churn rate by contract", "Which factors drive churn?",
                "Predict: new customer, fiber, month-to-month, electronic check",
                "How accurate is the model?"]
    cols = st.columns(len(examples))
    clicked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, width="stretch"):
            clicked = ex
    box = st.container()
    typed = st.chat_input("Ask a question or describe a customer…")
    q = typed or clicked
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        reply, mode = answer(q)
        st.session_state.messages.append({"role": "assistant", "content": reply, "mode": mode})
    with box:
        if not st.session_state.messages:
            st.markdown(HELP)
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("mode"):
                    st.caption("🤖 Gemini" if m["mode"] == "ai" else "⚙️ Offline mode")
