import tempfile
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from src.tools import load_resume_from_path
from src.agent import run_analysis
from src.job_finder_agent import run_job_finder

# --- Page config ---
st.set_page_config(
    page_title="Resume AI Career Advisor",
    page_icon="🧠",
    layout="wide"
)

# --- Header ---
st.title("🧠 Resume AI Career Advisor")
st.markdown(
    "Upload your resume and enter a target job role to get a **skill gap analysis** "
    "and discover **live job openings** with direct apply links."
)

st.divider()

# --- Shared Inputs ---
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "📄 Upload your Resume (PDF)",
        type=["pdf"],
        help="Only PDF files are supported."
    )
with col2:
    target_role = st.text_input(
        "🎯 Target Job Role",
        placeholder="e.g. Data Engineer",
    )

st.divider()

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Skill Gap Analysis", "🏢 Job Openings"])

# ──────────────────────────────────────────────
# TAB 1 — Skill Gap Analysis Agent
# ──────────────────────────────────────────────
with tab1:
    st.markdown("### Analyze your resume against real job requirements")
    analyze_btn = st.button(
        "🚀 Analyze My Resume", use_container_width=True,
        type="primary", key="analyze_btn"
    )

    if analyze_btn:
        if not uploaded_file:
            st.warning("Please upload your resume PDF.")
        elif not target_role.strip():
            st.warning("Please enter a target job role.")
        else:
            with st.spinner(f"Analyzing your resume against live **{target_role}** job postings..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    resume_text = load_resume_from_path(tmp_path)

                    if resume_text.startswith("Error"):
                        st.error(f"Could not read resume: {resume_text}")
                    else:
                        result = run_analysis(resume_text, target_role.strip())
                        st.success("Analysis complete!")
                        st.divider()
                        st.markdown(result)

                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")

# ──────────────────────────────────────────────
# TAB 2 — Job Finder Agent
# ──────────────────────────────────────────────
with tab2:
    st.markdown("### Find companies hiring for your target role right now")
    find_btn = st.button(
        "🔍 Find Job Openings", use_container_width=True,
        type="primary", key="find_btn"
    )

    if find_btn:
        if not uploaded_file:
            st.warning("Please upload your resume PDF.")
        elif not target_role.strip():
            st.warning("Please enter a target job role.")
        else:
            with st.spinner(f"Searching live **{target_role}** openings across LinkedIn, Indeed, Glassdoor..."):
                try:
                    # Re-read the file (Streamlit file pointer may have moved)
                    uploaded_file.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    resume_text = load_resume_from_path(tmp_path)

                    if resume_text.startswith("Error"):
                        st.error(f"Could not read resume: {resume_text}")
                    else:
                        ai_summary, job_list = run_job_finder(resume_text, target_role.strip())

                        st.success(f"Found **{len(job_list)}** live openings!")
                        st.divider()

                        # AI summary
                        st.markdown("#### 🤖 AI Match Summary")
                        st.markdown(ai_summary)
                        st.divider()

                        # Job cards
                        st.markdown(f"#### 📋 Live Openings — {target_role}")
                        if not job_list:
                            st.info("No openings found. Try a different role.")
                        else:
                            for job in job_list:
                                with st.container(border=True):
                                    left, right = st.columns([4, 1])
                                    with left:
                                        # Logo + title
                                        if job["logo"]:
                                            st.image(job["logo"], width=40)
                                        st.markdown(f"**{job['title']}**")
                                        st.markdown(f"🏢 {job['company']}")
                                        st.markdown(
                                            f"📍 {job['location']}  "
                                            f"{'🌐 Remote' if job['is_remote'] else ''}  "
                                            f"⏱ {job['employment_type']}"
                                        )
                                        st.caption(f"Posted: {job['posted'][:10] if job['posted'] != 'N/A' else 'N/A'}")
                                    with right:
                                        if job["apply_link"]:
                                            st.link_button(
                                                "Apply →",
                                                job["apply_link"],
                                                use_container_width=True,
                                                type="primary"
                                            )
                                        else:
                                            st.caption("No link available")

                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini · JSearch (RapidAPI) · LangGraph")
