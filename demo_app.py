"""
IncidentIQ - Interactive SRE Investigation & Triage Console
Autonomous Cloud Incident Investigation & Grounding Engine
Run with: streamlit run demo_app.py
"""
import time
import os
import streamlit as st
import pandas as pd

from incidentiq.pipeline import IncidentIQ
from incidentiq.evaluation.agent_metrics import evaluate_agent
from incidentiq.config import GEMINI_API_KEY, LANGFUSE_PUBLIC_KEY

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IncidentIQ — Autonomous Incident Triage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling: Modern Dark SRE Console Aesthetic
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 4px;
    }
    
    /* Evidence Pills */
    .evidence-pill {
        display: inline-block;
        background-color: #21262d;
        color: #79c0ff;
        border: 1px solid #388bfd40;
        padding: 2px 8px;
        border-radius: 6px;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.82em;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    
    .node-tag {
        display: inline-block;
        background-color: #1f242c;
        color: #ff7b72;
        border: 1px solid #ff7b7240;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85em;
        margin-right: 6px;
    }
    
    .status-badge-pass {
        background-color: #23863622;
        color: #3fb950;
        border: 1px solid #238636;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .status-badge-warn {
        background-color: #d2992222;
        color: #d29922;
        border: 1px solid #d29922;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Telemetry log box */
    .log-box {
        background-color: #0b0e14;
        border-left: 3px solid #58a6ff;
        border-radius: 4px;
        padding: 10px 14px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #c9d1d9;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Engine Resource Loader
# -----------------------------------------------------------------------------
DATA_PATH = "data/processed/logs.parquet"

@st.cache_resource(show_spinner="Initializing IncidentIQ Search Engine & Embeddings...")
def load_engine():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at `{DATA_PATH}`. Please check data directory.")
        return None
    return IncidentIQ(data_path=DATA_PATH)


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Preloaded Incidents
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚡ **IncidentIQ Console**")
st.sidebar.caption("Autonomous Incident Investigation & Grounding Engine")

# Observability Badges
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if GEMINI_API_KEY:
        st.caption("🟢 **Gemini**: Ready")
    else:
        st.caption("🔴 **Gemini**: Missing Key")
with col_s2:
    if LANGFUSE_PUBLIC_KEY:
        st.caption("🟢 **Langfuse**: Active")
    else:
        st.caption("⚪ **Langfuse**: Local")

st.sidebar.divider()

st.sidebar.markdown("### 🎯 **Select Incident Scenario**")
preloaded_incidents = {
    "Hardware Node Failure (Exact Path)": {
        "query": "R02-M1-N0-C:J12-U11",
        "desc": "Tests custom regex tokenizer & positional BM25 node recall"
    },
    "Kernel Architecture Crash (TLB Error)": {
        "query": "data TLB error interrupt",
        "desc": "Tests dense semantic / symptom retrieval for fatal CPU interrupts"
    },
    "Partial Rack Failure (Hybrid Fusion)": {
        "query": "node R02 critical failure",
        "desc": "Tests Reciprocal Rank Fusion (RRF) combining sparse and dense scores"
    },
    "Bus / DCR Timeout Alert": {
        "query": "MACHINE CHECK DCR read timeout",
        "desc": "Tests hardware bus machine check exception investigation"
    }
}

scenario_choice = st.sidebar.selectbox(
    "Preloaded SRE Incidents:",
    options=["Custom Incident Query"] + list(preloaded_incidents.keys()),
    index=1
)

if scenario_choice == "Custom Incident Query":
    query_input = st.sidebar.text_input(
        "Enter log message, symptom, or hardware node ID:",
        value="R02-M1-N0-C:J12-U11"
    )
    st.sidebar.caption("💡 Free-form search across system logs")
else:
    preset = preloaded_incidents[scenario_choice]
    query_input = st.sidebar.text_input(
        "Incident Query:",
        value=preset["query"]
    )
    st.sidebar.caption(f"ℹ️ *{preset['desc']}*")

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ **Retrieval & Pipeline Tuning**")
top_k = st.sidebar.slider("Initial Hybrid Top-K", min_value=5, max_value=30, value=10, step=5)

run_btn = st.sidebar.button(
    "Run Autonomous Investigation 🚀",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown("---")
with st.sidebar.expander("🏗️ Pipeline Architecture"):
    st.markdown("""
    **Core Pipeline Components:**
    1. **Positional & Inverted Index**: Custom BM25 with node regex extraction.
    2. **Dense Semantic Embedding**: Multi-layer transformer embeddings.
    3. **RRF Hybrid Scoring**: Sparse & dense rank merging without manual weight tuning.
    4. **Autonomous ReAct Agent**: Tool-augmented Gemini reasoning loop.
    5. **Provenance Auditor**: Zero-hallucination citation validation against retrieved evidence.
    """)

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("⚡ IncidentIQ — Autonomous Incident Triage Console")
st.markdown(
    "Real-time log telemetry synthesis, multi-stage hybrid retrieval (BM25 + Semantic + RRF), "
    "and **citation-audited root cause analysis** for production cloud/SRE outages."
)

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "investigation_result" not in st.session_state:
    st.session_state["investigation_result"] = None
if "investigation_latency" not in st.session_state:
    st.session_state["investigation_latency"] = None
if "investigation_metrics" not in st.session_state:
    st.session_state["investigation_metrics"] = None

# -----------------------------------------------------------------------------
# Run Investigation Handler
# -----------------------------------------------------------------------------
if run_btn:
    engine = load_engine()
    if engine is None:
        st.stop()

    with st.status("🔍 Investigating incident...", expanded=True) as status:
        st.write("1️⃣ Running sparse BM25 + dense semantic retrieval with Reciprocal Rank Fusion...")
        start_time = time.perf_counter()

        # Run full investigation
        result = engine.investigate(query=query_input, top_k=top_k)
        duration = time.perf_counter() - start_time

        st.write("2️⃣ Autonomous agent evaluated initial evidence and executed iterative tool queries...")
        st.write(f"   - Tool calls executed: **{len(result.tool_calls)}**")
        st.write(f"   - Tool evidence gathered: **{len(result.tool_evidence)} records**")

        st.write("3️⃣ Synthesized structured Root Cause Analysis and audited citation provenance...")
        metrics = evaluate_agent(result)

        # Store in session state
        st.session_state["investigation_result"] = result
        st.session_state["investigation_latency"] = duration
        st.session_state["investigation_metrics"] = metrics

        status.update(
            label=f"Investigation Complete ({duration:.2f}s) — Provenance Audited",
            state="complete",
            expanded=False
        )

# -----------------------------------------------------------------------------
# Render Results View
# -----------------------------------------------------------------------------
if st.session_state["investigation_result"] is not None:
    result = st.session_state["investigation_result"]
    latency = st.session_state["investigation_latency"]
    metrics = st.session_state["investigation_metrics"]
    analysis = result.analysis

    # -------------------------------------------------------------------------
    # Top KPI Metrics Dashboard
    # -------------------------------------------------------------------------
    st.markdown("### 📊 **Investigation Telemetry & Grounding Metrics**")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Citation Provenance</div>
            <div class="metric-value">{metrics['observation_citation_rate'] * 100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        is_grounded = metrics["grounded"]
        badge = "PASS ✅" if is_grounded else "AUDIT NEEDED ⚠️"
        val_color = "#3fb950" if is_grounded else "#d29922"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Hallucination Audit</div>
            <div class="metric-value" style="color: {val_color}; font-size: 1.4rem;">{badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        tool_eff = f"{metrics['tool_efficiency']:.1f} id/call" if metrics['tool_efficiency'] is not None else "Direct"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Tool Yield Efficiency</div>
            <div class="metric-value" style="font-size: 1.5rem;">{tool_eff}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Latency</div>
            <div class="metric-value">{latency:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        total_evidence_count = len(result.evidence) + len(result.tool_evidence)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Evidence</div>
            <div class="metric-value">{total_evidence_count} docs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Main Tabs: RCA, Evidence Inspector, Evaluation Audit, Raw Trace
    # -------------------------------------------------------------------------
    tab_rca, tab_evidence, tab_agent, tab_audit = st.tabs([
        "📋 Root Cause Analysis & Hypotheses",
        "📑 Retrieved Telemetry Evidence",
        "🛠️ Autonomous Agent Execution",
        "🛡️ Grounding & Provenance Audit"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Root Cause Analysis
    # -------------------------------------------------------------------------
    with tab_rca:
        col_summary, col_hypo = st.columns([1, 1], gap="medium")

        with col_summary:
            st.subheader("📌 Executive Incident Summary")
            st.info(analysis.summary)

            st.markdown("#### 🔬 Key Observed Signals")
            if analysis.observations:
                for idx, obs in enumerate(analysis.observations, 1):
                    with st.container():
                        st.markdown(f"**{idx}.** {obs.statement}")
                        if obs.evidence_ids:
                            pills = " ".join([f"<span class='evidence-pill'>Doc #{eid}</span>" for eid in obs.evidence_ids])
                            st.markdown(f"Grounding Citations: {pills}", unsafe_allow_html=True)
            else:
                st.caption("No specific observations recorded.")

            st.markdown("#### ❓ Unresolved Unknowns")
            if analysis.unknowns:
                for u in analysis.unknowns:
                    st.markdown(f"- ⚠️ {u}")
            else:
                st.markdown("✅ *No critical unknowns remaining. Root cause isolated.*")

        with col_hypo:
            st.subheader("💡 Formulated Hypotheses")
            if analysis.hypotheses:
                for idx, hyp in enumerate(analysis.hypotheses, 1):
                    conf_pct = int(hyp.confidence * 100)
                    with st.expander(f"Hypothesis {idx}: {hyp.statement[:65]}... ({conf_pct}% confidence)", expanded=True):
                        st.markdown(f"**Statement:** {hyp.statement}")
                        st.progress(hyp.confidence, text=f"Agent Confidence: {conf_pct}%")
                        if hyp.evidence_ids:
                            c_pills = " ".join([f"<span class='evidence-pill'>Doc #{eid}</span>" for eid in hyp.evidence_ids])
                            st.markdown(f"**Supporting Evidence:** {c_pills}", unsafe_allow_html=True)

            st.markdown("#### 🚀 Actionable Next Steps & Mitigations")
            if analysis.next_steps:
                for idx, step in enumerate(analysis.next_steps, 1):
                    with st.container():
                        st.markdown(f"**{idx}. {step.action}**")
                        st.caption(f"Reason: {step.reason}")
                        if step.query:
                            st.code(f"Suggested query: {step.query}", language="bash")
                        if step.evidence_ids:
                            step_pills = " ".join([f"<span class='evidence-pill'>Doc #{eid}</span>" for eid in step.evidence_ids])
                            st.markdown(f"Citations: {step_pills}", unsafe_allow_html=True)
            else:
                st.caption("Investigation complete. No further steps required.")

    # -------------------------------------------------------------------------
    # TAB 2: Grounding Telemetry Evidence
    # -------------------------------------------------------------------------
    with tab_evidence:
        st.subheader("📑 Telemetry Records & Provenance Context")
        st.caption("All log events retrieved through initial hybrid search (BM25 + Semantic) and agent tool expansions.")

        # Combine and deduplicate evidence
        all_evidence = []
        seen = set()

        for ev in result.evidence:
            doc_id = ev.get("doc_id")
            if doc_id not in seen:
                seen.add(doc_id)
                item = dict(ev)
                item["source"] = "Initial Hybrid Search"
                all_evidence.append(item)

        for ev in result.tool_evidence:
            doc_id = ev.get("doc_id")
            if doc_id not in seen:
                seen.add(doc_id)
                item = dict(ev)
                item["source"] = "Autonomous Tool Call"
                all_evidence.append(item)

        # Filter bar
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            search_filter = st.text_input("Filter evidence records by keyword or node ID:", value="")
        with filter_col2:
            source_filter = st.selectbox("Filter by Source:", ["All Records", "Initial Hybrid Search", "Autonomous Tool Call"])

        filtered_evidence = all_evidence
        if search_filter:
            filtered_evidence = [
                e for e in filtered_evidence
                if search_filter.lower() in str(e.get("message", "")).lower()
                or search_filter.lower() in str(e.get("node", "")).lower()
                or search_filter.lower() in str(e.get("doc_id", "")).lower()
            ]
        if source_filter != "All Records":
            filtered_evidence = [e for e in filtered_evidence if e.get("source") == source_filter]

        st.markdown(f"Showing **{len(filtered_evidence)}** of **{len(all_evidence)}** total retrieved log events.")

        for item in filtered_evidence:
            doc_id = item.get("doc_id")
            node = item.get("node", "N/A")
            severity = item.get("severity", "N/A")
            timestamp = item.get("timestamp", "N/A")
            source = item.get("source", "Initial")
            msg = item.get("message", "")

            source_badge = "🔵 Hybrid Search" if "Initial" in source else "🟣 Agent Tool Discovery"

            st.markdown(f"""
            <div class="log-box">
                <span class="evidence-pill">Doc #{doc_id}</span>
                <span class="node-tag">Node: {node}</span>
                <span style="color: #8b949e; font-size: 0.8rem; margin-right: 8px;">Severity: <b>{severity}</b></span>
                <span style="color: #8b949e; font-size: 0.8rem; margin-right: 8px;">Time: {timestamp}</span>
                <span style="float: right; font-size: 0.8rem; color: #58a6ff;">{source_badge}</span>
                <div style="margin-top: 6px; color: #e6edf3; font-family: monospace;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 3: Autonomous Agent Execution & Tracing
    # -------------------------------------------------------------------------
    with tab_agent:
        st.subheader("🛠️ Agent Reasoning & Tool Invocation Trace")
        st.markdown(
            "IncidentIQ's agent reasons iteratively over retrieved evidence, identifies unknowns, "
            "and automatically invokes search tools to follow causal leads."
        )

        agent_col1, agent_col2 = st.columns([1, 1])

        with agent_col1:
            st.markdown("#### 🔄 Iterative Search Tool Calls")
            if result.tool_calls:
                for idx, call_str in enumerate(result.tool_calls, 1):
                    with st.expander(f"Tool Call #{idx}", expanded=True):
                        st.code(call_str, language="json")
            else:
                st.info("Direct investigation: The initial hybrid retrieval provided sufficient grounding evidence without requiring additional iterations.")

        with agent_col2:
            st.markdown("#### 🛰️ Observability & Tracing")
            st.markdown("""
            This investigation is fully instrumented with **Langfuse Tracing**:
            - **End-to-End Latency Tracking**: Sparse retrieval, dense embedding, LLM reasoning steps.
            - **Token Economics**: Input/output token usage per iteration.
            - **Step-level Reproducibility**: Input state, context payload, generated hypotheses.
            """)
            if LANGFUSE_PUBLIC_KEY:
                st.success("🟢 Langfuse Observability is active. View real-time traces in your Langfuse dashboard.")
            else:
                st.info("To view cloud traces, configure `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env`.")

    # -------------------------------------------------------------------------
    # TAB 4: Grounding & Provenance Audit
    # -------------------------------------------------------------------------
    with tab_audit:
        st.subheader("🛡️ Quantitative Grounding & Hallucination Defense")
        st.markdown(
            "Every statement generated by the agent must be cited back to real retrieved document IDs. "
            "IncidentIQ enforces mathematical citation integrity to eliminate AI hallucinations."
        )

        audit_col1, audit_col2 = st.columns(2)

        with audit_col1:
            st.markdown("#### 📋 Integrity Breakdown")
            audit_df = pd.DataFrame([
                {"Metric": "Observation Citation Rate", "Score": f"{metrics['observation_citation_rate'] * 100:.1f}%", "Status": "PASS ✅" if metrics['observation_citation_rate'] >= 0.8 else "REVIEW ⚠️"},
                {"Metric": "Hypothesis Citation Rate", "Score": f"{metrics['hypothesis_citation_rate'] * 100:.1f}%", "Status": "PASS ✅" if metrics['hypothesis_citation_rate'] >= 0.8 else "REVIEW ⚠️"},
                {"Metric": "Next Step Citation Rate", "Score": f"{metrics['next_step_citation_rate'] * 100:.1f}%", "Status": "PASS ✅" if metrics['next_step_citation_rate'] >= 0.8 else "REVIEW ⚠️"},
                {"Metric": "Completion Consistency", "Score": str(metrics['completion_consistent']), "Status": "PASS ✅" if metrics['completion_consistent'] else "FAIL ❌"},
                {"Metric": "Grounded Zero-Hallucination", "Score": str(metrics['grounded']), "Status": "PASS ✅" if metrics['grounded'] else "FAIL ❌"},
            ])
            st.table(audit_df)

        with audit_col2:
            st.markdown("#### 🔍 Citation Verification Details")
            violations = metrics.get("citation_violations", [])
            missing = metrics.get("missing_citations", [])

            if not violations and not missing:
                st.success("🎉 **Zero Citations Violations Detected!** All cited document IDs exist in the retrieved evidence corpus.")
            else:
                if violations:
                    st.warning(f"⚠️ {len(violations)} Invalid Citation(s) Found (Hallucinated IDs):")
                    for v in violations:
                        st.json(v)
                if missing:
                    st.info(f"ℹ️ {len(missing)} Claim(s) without Citations:")
                    for m in missing:
                        st.caption(f"- Section: `{m.get('section')}` | Item: `{m.get('item_index')}`: {m.get('reason')}")

else:
    # -------------------------------------------------------------------------
    # Empty State: Welcome Banner & Quick Start
    # -------------------------------------------------------------------------
    st.info("👈 **Select an incident scenario from the sidebar** and click **Run Autonomous Investigation 🚀** to begin.")

    st.markdown("### 🌟 Why IncidentIQ Outperforms Standard RAG")
    card_col1, card_col2, card_col3 = st.columns(3)

    with card_col1:
        st.markdown("""
        #### 1. Custom BM25 + Regex
        Standard vector search fails on precise hardware paths like `R02-M1-N0-C:J12-U11`. Our custom inverted index preserves structural hardware identifiers with zero token truncation.
        """)

    with card_col2:
        st.markdown("""
        #### 2. Reciprocal Rank Fusion
        Combines lexical keyword precision with dense transformer semantics. No arbitrary weighting hyperparameters; rank-based merging guarantees optimal top-k recall.
        """)

    with card_col3:
        st.markdown("""
        #### 3. Audited Grounding
        Prevents hallucinated SRE advice. Every observation and hypothesis is audited against retrieved evidence IDs, computing a quantitative provenance rate.
        """)

    st.markdown("---")
    st.markdown("#### 🧪 Test Scenarios Included:")
    for name, data in preloaded_incidents.items():
        st.markdown(f"- **{name}**: `{data['query']}` — *{data['desc']}*")
