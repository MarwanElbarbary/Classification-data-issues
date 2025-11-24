import streamlit as st
import pandas as pd
from transformers import pipeline
import zipfile
import io

st.set_page_config(
    page_title="Issue Prioritization Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="distilbert-base-uncased", device=-1)

try:
    with st.spinner("Loading model..."):
        classifier = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def prioritize_issues(df, text_column):
    df = df.copy()
    scores = []
    progress_bar = st.progress(0)
    total = len(df)
    for idx, row in df.iterrows():
        text = str(row[text_column])
        try:
            result = classifier(text)
            score = result[0]["score"] if result else 0.0
        except Exception:
            score = 0.0
        scores.append(score)
        progress_bar.progress((idx + 1) / total)
    progress_bar.empty()
    df["priority_score"] = scores

    grouped = (
        df.groupby(text_column, as_index=False)["priority_score"]
        .max()
        .sort_values(by="priority_score", ascending=False)
        .reset_index(drop=True)
    )
    counts = (
        df.groupby(text_column)
        .size()
        .reset_index(name="occurrences")
    )
    result = grouped.merge(counts, on=text_column, how="left")
    result = result.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
    return result

# ---------- GLOBAL STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}
.stApp {
    background: linear-gradient(145deg, #0f172a, #020617);
    color: #e5e7eb;
}
.app-header {
    padding: 1.5rem 1rem 1rem 1rem;
    border-radius: 15px;
    background: #1f2937;
    border: 1px solid #374151;
    margin-bottom: 1.5rem;
    color: #e5e7eb;
}
.app-header h1 {
    margin: 0;
    font-size: 2rem;
    color: #f9fafb;
}
.app-header p {
    margin: 0.3rem 0 0 0;
    color: #9ca3af;
    font-size: 0.95rem;
}
.metric-card {
    padding: 1rem 1.2rem;
    border-radius: 15px;
    background: #111827;
    border: 1px solid #374151;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}
.metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e5e7eb;
}
.stButton>button {
    border-radius: 12px;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s;
}
.stButton>button:hover {
    filter: brightness(1.1);
}
.dataframe td, .dataframe th {
    color: #e5e7eb !important;
    background-color: #1f2937 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="app-header">
    <h1>Issue Prioritization Dashboard</h1>
    <p>Analyze issue data and rank critical items using AI text classification.</p>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_results, tab_config = st.tabs(["Upload & Settings", "Results", "Configuration"])
uploaded_file = None
df = None
ranked_df = None
selected_column = None

# ---------- TAB 1: UPLOAD & SETTINGS ----------
with tab_upload:
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV or ZIP containing a CSV", type=["csv", "zip"])

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
            selected_column = st.selectbox("Select the text column", column_names)

            col_left, col_right = st.columns(2)
            with col_left:
                sample_mode = st.checkbox("Quick sample (first 100 rows)", value=True)
            with col_right:
                top_n_default = 20 if len(df) > 20 else len(df)
                top_n = st.number_input("Rows to display", min_value=5, max_value=200, value=top_n_default, step=5)

            st.session_state["top_n"] = top_n
            st.session_state["sample_mode"] = sample_mode
            st.session_state["selected_column"] = selected_column

            if st.button("Run Prioritization"):
                work_df = df.head(100) if sample_mode else df
                with st.spinner("Running AI model on issues..."):
                    ranked_df = prioritize_issues(work_df, text_column=selected_column)
                st.session_state["ranked_df"] = ranked_df

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload a CSV or ZIP to start.")

# ---------- TAB 2: RESULTS ----------
with tab_results:
    st.subheader("Analysis & Dashboard")
    if "ranked_df" in st.session_state and "selected_column" in st.session_state:
        ranked_df = st.session_state["ranked_df"]
        selected_column = st.session_state["selected_column"]
        top_n = st.session_state.get("top_n", 20)

        col_filter_1, col_filter_2 = st.columns(2)
        with col_filter_1:
            min_priority = st.slider("Minimum priority score", 0.0, 1.0, 0.0, 0.01)
        with col_filter_2:
            min_occ = st.number_input("Minimum occurrences", 1, int(ranked_df["occurrences"].max()), 1)

        filtered_df = ranked_df[(ranked_df["priority_score"] >= min_priority) & (ranked_df["occurrences"] >= min_occ)]

        total_unique = len(filtered_df)
        top_priority = filtered_df["priority_score"].max() if total_unique > 0 else 0
        avg_priority = filtered_df["priority_score"].mean() if total_unique > 0 else 0
        total_occurrences = filtered_df["occurrences"].sum() if "occurrences" in filtered_df.columns else 0

        col_a, col_b, col_c, col_d = st.columns(4)
        metrics = [
            ("Unique issues", total_unique),
            ("Total records", total_occurrences),
            ("Highest priority score", round(top_priority,3)),
            ("Average priority score", round(avg_priority,3))
        ]
        for col, (label, value) in zip([col_a, col_b, col_c, col_d], metrics):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("Top Issues Table")
        st.dataframe(filtered_df[[selected_column, "priority_score", "occurrences"]].head(top_n))

        st.markdown("Top Issues Chart")
        if len(filtered_df) > 0:
            try:
                chart_df = filtered_df.head(top_n).set_index(selected_column)[["priority_score"]]
                st.bar_chart(chart_df)
            except:
                st.warning("Could not render chart.")

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered results CSV", csv_data, "issues_prioritized.csv", "text/csv")
    else:
        st.info("No results yet. Run prioritization from the first tab.")

# ---------- TAB 3: CONFIG ----------
with tab_config:
    st.subheader("Model & Run Information")
    st.markdown("""
    **Model:** distilbert-base-uncased  
    **Task:** Text Classification  
    **Details:**  
    - Each issue text is processed by the AI model.  
    - Model outputs a confidence score used as the priority indicator.  
    - Issues are grouped by text; max score is considered.  
    - Occurrences count duplicates in the dataset (full or sampled).  
    """)
    st.markdown("---")
    st.markdown("**Current Run Settings**")
    if "sample_mode" in st.session_state and "selected_column" in st.session_state:
        st.write(f"Text column: `{st.session_state['selected_column']}`")
        st.write(f"Sampling: {'First 100 rows' if st.session_state['sample_mode'] else 'Full file'}")
    else:
        st.write("No run executed yet.")
