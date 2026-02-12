import streamlit as st
import pandas as pd
from transformers import pipeline
import zipfile
from datetime import datetime

st.set_page_config(
    page_title="Issue Prioritization Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

try:
    with st.spinner("Loading AI model..."):
        classifier = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def score_to_priority_label(score):
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

def prioritize_issues(df, text_column):
    df = df.copy()

    texts = df[text_column].astype(str).fillna("").tolist()
    texts = [t[:512] for t in texts]

    progress_bar = st.progress(0)
    total = len(texts)

    try:
        results = classifier(texts, batch_size=32, truncation=True)
        scores = [r["score"] if r else 0.0 for r in results]
    except Exception:
        scores = [0.0] * total

    progress_bar.progress(1.0)
    progress_bar.empty()

    df["priority_score"] = scores
    df["priority_level"] = df["priority_score"].apply(score_to_priority_label)
    df["issue_ar"] = df[text_column]
    df["occurrences"] = 1

    grouped = (
        df.groupby("issue_ar", as_index=False)
        .agg({
            "priority_score": "max",
            "priority_level": lambda x: x.iloc[x.argmax()],
            "occurrences": "sum"
        })
        .sort_values(by="priority_score", ascending=False)
        .reset_index(drop=True)
    )

    return grouped

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at top left, #1d283a 0, #020617 52%, #020617 100%); color: #e5e7eb; }
[data-testid="stSidebar"] { background: #020617; border-right: 1px solid #1f2937; }
[data-testid="stSidebar"] h2 { font-size: 1.4rem; font-weight: 600; }
.app-header { padding: 1.2rem 1.5rem; border-radius: 18px; background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.9)); border: 1px solid rgba(55,65,81,0.8); margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem; box-shadow: 0 18px 45px rgba(0,0,0,0.65); }
.app-header-left { display: flex; flex-direction: column; gap: 0.25rem; }
.app-header-title { font-size: 1.9rem; font-weight: 600; color: #f9fafb; display: flex; align-items: center; gap: 0.4rem; }
.app-header-title span.icon { font-size: 1.9rem; }
.app-header-subtitle { font-size: 0.95rem; color: #9ca3af; }
.app-header-badge { padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; border: 1px solid #4b5563; color: #e5e7eb; background: radial-gradient(circle at top left, rgba(99,102,241,0.25), rgba(15,23,42,0.9)); }
.app-header-pill { padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; background: linear-gradient(90deg, rgba(96,165,250,0.25), rgba(129,140,248,0.2)); color: #c7d2fe; }
.metric-card { padding: 0.9rem 1.1rem; border-radius: 16px; background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(17,24,39,0.98)); border: 1px solid rgba(55,65,81,0.8); text-align: left; position: relative; overflow: hidden; }
.metric-label { font-size: 0.8rem; color: #9ca3af; margin-bottom: 0.3rem; }
.metric-value { font-size: 1.35rem; font-weight: 600; color: #e5e7eb; }
.stButton>button { border-radius: 999px; background: linear-gradient(90deg, #6366f1, #8b5cf6); color: white; font-weight: 600; padding: 0.55rem 1.4rem; border: none; box-shadow: 0 10px 30px rgba(79,70,229,0.45); }
.dataframe td, .dataframe th { color: #e5e7eb !important; background-color: #020617 !important; border-color: #111827 !important; font-size: 0.85rem !important; }
.panel { background: rgba(15,23,42,0.94); border-radius: 16px; border: 1px solid #1f2937; padding: 1rem 1.2rem; }
.panel-header { font-size: 0.9rem; font-weight: 500; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 Issue Prioritizer")
    st.markdown("Discover the most critical issues in your dataset using AI analysis.")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + Transformers")

st.markdown("""
<div class="app-header">
    <div class="app-header-left">
        <div class="app-header-title"><span class="icon">📊</span>Issue Prioritization Dashboard</div>
        <div class="app-header-subtitle">Upload dataset, apply AI scoring, and focus on critical issues.</div>
        <div class="app-header-pill">Prioritize by impact · Group duplicates · Filter by severity</div>
    </div>
    <div class="app-header-right">
        <span class="app-header-badge">⚙️ Powered by Hugging Face Sentiment Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_results, tab_config = st.tabs(["Upload & Settings", "Results", "Configuration"])

# ---------------- UPLOAD TAB ----------------
with tab_upload:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="panel"><div class="panel-header">Data & Settings</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV or ZIP", type=["csv", "zip"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".zip"):
                    with zipfile.ZipFile(uploaded_file) as z:
                        file_name = z.namelist()[0]
                        with z.open(file_name) as f:
                            df = pd.read_csv(f)
                else:
                    df = pd.read_csv(uploaded_file)

                st.success(f"File loaded successfully. Rows: {len(df)}")

                column_names = df.columns.tolist()
                selected_column = st.selectbox("Select text column", column_names)

                sample_option = st.selectbox(
                    "Rows to analyze",
                    ["First 100", "First 500", "First 1000", "Full dataset"]
                )

                top_n_default = 20 if len(df) > 20 else len(df)
                top_n = st.number_input("Max rows to display", min_value=5, max_value=200, value=top_n_default, step=5)

                if st.button("🚀 Run AI Prioritization"):
                    if sample_option == "First 100":
                        work_df = df.head(100)
                    elif sample_option == "First 500":
                        work_df = df.head(500)
                    elif sample_option == "First 1000":
                        work_df = df.head(1000)
                    else:
                        work_df = df

                    with st.spinner("Analyzing issues..."):
                        ranked_df = prioritize_issues(work_df, text_column=selected_column)
                        st.session_state["ranked_df"] = ranked_df
                        st.session_state["top_n"] = top_n

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error reading file: {e}")

    with right:
        st.markdown('<div class="panel"><div class="panel-header">Preview</div>', unsafe_allow_html=True)
        if uploaded_file:
            st.dataframe(df.head(5), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RESULTS TAB ----------------
with tab_results:
    st.markdown('<div class="panel"><div class="panel-header">Analysis & Dashboard</div>', unsafe_allow_html=True)

    if "ranked_df" in st.session_state:
        ranked_df = st.session_state["ranked_df"]
        top_n = st.session_state.get("top_n", 20)

        f1, f2, f3 = st.columns([1.1,1.1,1])
        with f1:
            min_priority = st.slider("Minimum priority score", 0.0, 1.0, 0.0, 0.01)
        with f2:
            min_occ = st.number_input("Minimum occurrences", 1, int(ranked_df["occurrences"].max()), 1)
        with f3:
            search_term = st.text_input("Search issues (optional)")

        filtered_df = ranked_df[(ranked_df["priority_score"] >= min_priority) & (ranked_df["occurrences"] >= min_occ)]

        if search_term:
            filtered_df = filtered_df[filtered_df["issue_ar"].astype(str).str.contains(search_term, case=False, na=False)]

        total_unique = len(filtered_df)
        top_priority = filtered_df["priority_score"].max() if total_unique > 0 else 0
        avg_priority = filtered_df["priority_score"].mean() if total_unique > 0 else 0
        total_occurrences = filtered_df["occurrences"].sum()

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Unique issues", total_unique),
            ("Total records", total_occurrences),
            ("Highest priority score", round(top_priority,3)),
            ("Average priority score", round(avg_priority,3))
        ]

        for col, (label, value) in zip([c1,c2,c3,c4], metrics):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Top Issues")

        display_df = filtered_df[["issue_ar", "priority_level", "priority_score", "occurrences"]].head(top_n).copy()
        display_df["priority_score"] = display_df["priority_score"].round(3)

        st.dataframe(display_df, use_container_width=True)

        st.markdown("### Priority Score Chart")
        chart_df = filtered_df.head(top_n).set_index("issue_ar")[["priority_score"]]
        st.bar_chart(chart_df)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download CSV", csv_data, f"issues_prioritized_{ts}.csv", "text/csv")

    else:
        st.info("No results yet. Run AI prioritization from 'Upload & Settings'.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- CONFIG TAB ----------------
with tab_config:
    st.markdown('<div class="panel"><div class="panel-header">Model & System Configuration</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 🤖 AI Model Overview

    **Model Used:** Hugging Face `sentiment-analysis` pipeline  
    **Architecture:** Transformer-based deep learning model  
    **Task Type:** Text Classification  

    The system analyzes each issue description using a pre-trained NLP model.
    The model evaluates the emotional intensity of the text and returns a confidence score.
    This confidence score is then converted into a priority level.

    ---
    """)

    st.markdown("""
    ## ⚙️ How Priority Is Calculated

    Each issue goes through the following steps:

    1. The text is truncated to 512 characters (model limit).
    2. The AI model assigns a confidence score between **0.0 and 1.0**.
    3. The score is mapped into a priority level:

       - **High** → Score ≥ 0.80  
       - **Medium** → Score ≥ 0.50  
       - **Low** → Score < 0.50  

    The higher the score, the more critical the issue is considered.

    ---
    """)

    st.markdown("""
    ## 📊 Data Processing Logic

    - Duplicate issues are grouped together.
    - The **maximum priority score** within duplicates is kept.
    - The system counts how many times each issue appears.
    - Results are sorted from highest to lowest priority.

    This helps decision makers:
    - Focus on the most impactful issues
    - Identify recurring problems
    - Filter by severity or frequency

    ---
    """)

    st.markdown("""
    ## 🚀 Performance Optimization

    - Batch inference is used to process multiple texts at once.
    - Model is cached using `st.cache_resource` to avoid reloading.
    - No external translation APIs are used (faster execution).
    - Results are stored in session state for fast dashboard rendering.

    ---
    """)

    if "ranked_df" in st.session_state:
        ranked_df = st.session_state["ranked_df"]

        st.markdown("## 📈 Current Run Statistics")

        total_unique = len(ranked_df)
        top_priority = ranked_df["priority_score"].max() if total_unique > 0 else 0
        avg_priority = ranked_df["priority_score"].mean() if total_unique > 0 else 0
        total_occurrences = ranked_df["occurrences"].sum()

        c1, c2, c3, c4 = st.columns(4)

        stats = [
            ("Unique Issues", total_unique),
            ("Total Occurrences", total_occurrences),
            ("Highest Priority Score", round(top_priority, 3)),
            ("Average Priority Score", round(avg_priority, 3))
        ]

        for col, (label, value) in zip([c1, c2, c3, c4], stats):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run the AI prioritization to view system statistics.")

    st.markdown("""
    ---
    ### 📌 Summary

    This dashboard transforms raw issue data into actionable insights using AI-powered text analysis.
    It enables teams to prioritize smarter, respond faster, and allocate resources efficiently.
    """)

    st.markdown('</div>', unsafe_allow_html=True)
