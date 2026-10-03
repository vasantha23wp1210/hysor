import os
import sys
import time
import pandas as pd
import streamlit as st

# App lives in src/ — so CURRENT_DIR IS the src folder already
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/
PROJECT_DIR = os.path.dirname(CURRENT_DIR)                 # .../uatie-edm-phase1/

# Add both src/ and project root to path
for _p in [CURRENT_DIR, PROJECT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# rag/ sits directly inside src/
RAG_DIR = os.path.join(CURRENT_DIR, "rag")
if os.path.isdir(RAG_DIR) and RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

# ── Imports (get_dynamic_recommendation removed — ICAO layer out of scope) ──
try:
    from rag.rag_query_engine import (
        initialize_resources,
        semantic_search,
        hybrid_search,
        ontology_nl_search,
        ontology_search,
        compute_risk_score,
        RISK_ICONS,
        INTENT_SLUG_MAP,
        TOP_K,
    )
    import rag.rag_query_engine as rag_engine
except Exception:
    try:
        from rag_query_engine import (
            initialize_resources,
            semantic_search,
            hybrid_search,
            ontology_nl_search,
            ontology_search,
            compute_risk_score,
            RISK_ICONS,
            INTENT_SLUG_MAP,
            TOP_K,
        )
        import rag.rag_query_engine as rag_engine
    except Exception:
        from rag.rag_query_engine import (
            initialize_resources,
            semantic_search,
            hybrid_search,
            ontology_nl_search,
            ontology_search,
            compute_risk_score,
            RISK_ICONS,
            INTENT_SLUG_MAP,
            TOP_K,
        )
        import rag.rag_query_engine as rag_engine

st.set_page_config(
    page_title="HySOR — Aviation Safety Incident Retrieval",
    page_icon="✈️",
    layout="wide",
)

@st.cache_resource
def load_backend():
    initialize_resources(verbose=False)
    return True

load_backend()


def safe_results_for_df(results):
    """Build display DataFrame — phraseme_link excluded."""
    rows = []
    for r in results:
        rows.append({
            "rank":            r.get("rank", ""),
            "score":           r.get("score", 0.0),
            "recency_score":   r.get("recency_score", 0.0),
            "composite_score": r.get("composite_score", 0.0),
            "year":            r.get("year", "Unknown"),
            "month":           r.get("month", "Unknown"),
            "intent":          r.get("intent", ""),
            "phase":           r.get("phase", ""),
            "hazards":         r.get("hazards", ""),
            "source":          r.get("source", ""),
            "incident_id":     r.get("incident_id", ""),
            "text":            r.get("text", ""),
        })
    return pd.DataFrame(rows)


def run_query(query, mode, topk):
    if not query.strip():
        return [], None
    if mode == "semantic" and rag_engine.SEMANTIC_OK:
        return semantic_search(query, topk), None
    if mode == "hybrid" and rag_engine.SEMANTIC_OK:
        return hybrid_search(query, topk), None
    matched_intent, results = ontology_nl_search(query, topk)
    return results, matched_intent


def get_top_intent_slug(results):
    if not results:
        return None
    return results[0].get("intent", "").replace(" ", "_").upper()


# ── Page header ───────────────────────────────────────────────────────────────
st.title("✈ HySOR — Aviation Safety Incident Retrieval")
st.caption("Hybrid Semantic-Ontological Retrieval · FAISS + RDF/OWL · Recency-aware ranking")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Search Controls")
mode       = st.sidebar.selectbox("Mode", ["hybrid", "semantic", "ontology"], index=0)
topk       = st.sidebar.slider("Top-K results", min_value=1, max_value=20, value=min(TOP_K, 20))
show_raw   = st.sidebar.checkbox("Show raw result JSON", value=False)
show_table = st.sidebar.checkbox("Show results table", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Intent shortcuts")
intent_choice = st.sidebar.selectbox(
    "Ontology intent lookup", ["-- none --"] + list(INTENT_SLUG_MAP.keys())
)
intent_go = st.sidebar.button("Run intent lookup")

st.sidebar.markdown("---")
st.sidebar.subheader("System status")
st.sidebar.success(f"Semantic ready: {rag_engine.SEMANTIC_OK}")
st.sidebar.success(f"Ontology ready: {rag_engine.ONTO_OK}")
st.sidebar.info(f"Top-K default: {TOP_K}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Search", "Corpus Metrics"])

with tab1:
    query = st.text_input("Enter query", placeholder="pilot failed to read back clearance")
    col_a, col_b = st.columns([1, 5])
    with col_a:
        search_clicked = st.button("Search", use_container_width=True)
    with col_b:
        st.markdown(
            "Try: `late runway change during approach`, "
            "`windshear on final`, `engine failure during takeoff roll`"
        )

    results       = []
    matched_intent = None

    if intent_go and intent_choice != "-- none --":
        slug = INTENT_SLUG_MAP[intent_choice]
        with st.spinner(f"Running ontology lookup for {slug}..."):
            start   = time.perf_counter()
            results = ontology_search(slug, topk)
            elapsed = time.perf_counter() - start
        st.success(f"Found {len(results)} results in {elapsed:.3f} sec for intent {slug}")

    elif search_clicked and query.strip():
        with st.spinner("Searching..."):
            start = time.perf_counter()
            results, matched_intent = run_query(query, mode, topk)
            elapsed = time.perf_counter() - start
        st.success(f"Found {len(results)} results in {elapsed:.3f} sec")
        if matched_intent:
            st.info(f"Ontology keyword intent matched: {matched_intent}")

    if results:
        for i, r in enumerate(results, 1):
            r["rank"] = i

        # ── Results table ──────────────────────────────────────────────────
        if show_table:
            st.subheader("Results table")
            st.dataframe(
                safe_results_for_df(results),
                use_container_width=True,
                hide_index=True,
            )

        # ── Detailed result cards ──────────────────────────────────────────
        st.subheader("Detailed results")
        for r in results:
            risk = compute_risk_score(
                r.get("intent",  "Unknown"),
                r.get("phase",   "Unknown"),
                r.get("hazards", ""),
            )
            icon   = RISK_ICONS.get(risk, "")
            header = (
                f"[{r.get('rank', '?')}] {r.get('intent', '?')} | "
                f"Sim={r.get('score', 0.0):.4f} | "
                f"Rec={r.get('recency_score', 0.0):.4f} | "
                f"Comp={r.get('composite_score', 0.0):.4f} | "
                f"Risk={icon} {risk}"
            )
            with st.expander(header):
                st.write(r.get("text", ""))
                st.caption(
                    f"Hazards: {r.get('hazards', '')} | "
                    f"Source: {r.get('source', '')} | "
                    f"Incident ID: {r.get('incident_id', '—')} | "
                    f"Year: {r.get('year', 'Unknown')} | "
                    f"Month: {r.get('month', 'Unknown')}"
                )
                if show_raw:
                    st.json(r)

    elif search_clicked or intent_go:
        st.warning("No results found.")

# ── Corpus Metrics tab ────────────────────────────────────────────────────────
with tab2:
    if getattr(rag_engine, "df_full", None) is not None:
        total         = len(rag_engine.df_full)
        classified    = (
            (rag_engine.df_full["intent"] != "Unclassified").sum()
            if "intent" in rag_engine.df_full.columns else 0
        )
        unclassified  = total - classified
        classified_pct = (classified / total * 100) if total else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{total:,}")
        col2.metric("Classified",    f"{classified:,}")
        col3.metric("Unclassified",  f"{unclassified:,}")
        col4.metric("Classified %",  f"{classified_pct:.1f}%")

        if "intent" in rag_engine.df_full.columns:
            st.subheader("Intent distribution")
            st.bar_chart(
                rag_engine.df_full["intent"]
                .fillna("Unknown").value_counts().head(15)
            )

        if "phase_final" in rag_engine.df_full.columns:
            st.subheader("Phase-of-flight distribution")
            st.bar_chart(
                rag_engine.df_full["phase_final"]
                .fillna("Unknown").value_counts().head(15)
            )
    else:
        st.warning("Annotated corpus is not loaded.")
