import os
import sys
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure local services package is importable first
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import importlib

try:
    import services.ai_service as ai_service_module
    importlib.reload(ai_service_module)
    from services.document_processor import extract_document_text, DocumentProcessingError
    from services.ai_service import (
        analyze_document_with_ai,
        ask_document_ai,
        is_groq_configured,
        get_groq_api_key,
        get_demo_analysis
    )
    from services.ml_service import assess_document_ml
except ImportError:
    import ai_service as ai_service_module
    importlib.reload(ai_service_module)
    from document_processor import extract_document_text, DocumentProcessingError
    from ai_service import (
        analyze_document_with_ai,
        ask_document_ai,
        is_groq_configured,
        get_groq_api_key,
        get_demo_analysis
    )
    from ml_service import assess_document_ml

# Load environment dynamically
env_file_path = Path(__file__).resolve().parent / ".env"
if env_file_path.exists():
    load_dotenv(dotenv_path=env_file_path, override=True)
else:
    load_dotenv(override=True)

# Page Configuration
st.set_page_config(
    page_title="FormFix AI | Don't just understand the form. Know what to do next.",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    /* Hide Streamlit Sidebar Completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Global styles */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .tagline {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #4B5563;
        max-width: 780px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }
    
    /* Feature cards on Landing Page */
    .step-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        height: 100%;
        text-align: left;
    }
    .step-number {
        font-size: 1.5rem;
        font-weight: 800;
        color: #2563EB;
        margin-bottom: 0.3rem;
    }
    .step-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.3rem;
    }
    .step-text {
        font-size: 0.92rem;
        color: #4B5563;
        line-height: 1.45;
    }

    /* Dashboard result cards */
    .hero-action-card {
        background: linear-gradient(180deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #86EFAC;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .hero-action-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #166534;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .deadline-card {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #FCA5A5;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .deadline-title {
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #991B1B;
    }
    .deadline-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #B91C1C;
        margin-top: 0.2rem;
    }

    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .field-card {
        background: #F9FAFB;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .field-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .field-desc {
        font-size: 0.93rem;
        color: #374151;
        margin: 0.25rem 0;
    }
    .field-doc {
        font-size: 0.85rem;
        font-weight: 600;
        color: #047857;
    }

    .warning-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "checklist_state" not in st.session_state:
    st.session_state.checklist_state = {}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "is_demo_mode" not in st.session_state:
    st.session_state.is_demo_mode = False
if "custom_groq_key" not in st.session_state:
    st.session_state.custom_groq_key = ""
if "ml_assessment" not in st.session_state:
    st.session_state.ml_assessment = None


# Top bar with Language Selector
col_brand, col_lang = st.columns([4, 1.3])
with col_lang:
    selected_lang_choice = st.selectbox(
        "🌐 Language / भाषा",
        ["English", "हिन्दी (Hindi)", "मराठी (Marathi)"],
        index=0,
        key="app_lang_choice"
    )

LANG_MAP = {
    "English": "English",
    "हिन्दी (Hindi)": "Hindi",
    "मराठी (Marathi)": "Marathi"
}
target_language = LANG_MAP[selected_lang_choice]
st.session_state.target_language = target_language

# Header / Landing Hero
with col_brand:
    st.markdown("""
    <div class="main-header" style="margin-bottom: 0.5rem; text-align: left;">
        <div class="main-title">FormFix AI</div>
        <div class="tagline">Don’t just understand the form. Know what to do next.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero-desc" style="margin-bottom: 1.5rem;">
    Upload a complicated form, notice, circular, or official document and let AI turn it into simple explanations, prominent deadlines, required documents, and an actionable step-by-step checklist.
</div>
""", unsafe_allow_html=True)

# 3-Step Process Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">01</div>
        <div class="step-title">Upload</div>
        <div class="step-text">Upload your college notice, scholarship form, circular PDF, or scanned document image.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">02</div>
        <div class="step-title">Understand</div>
        <div class="step-text">AI extracts clear facts, decodes confusing bureaucratic fields, and highlights critical deadlines.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">03</div>
        <div class="step-title">Take Action</div>
        <div class="step-text">Get an interactive, prioritized checklist of every single step and supporting document needed.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Document Upload Section (Shown when no analysis or when user wants to analyze)
with st.container():
    st.markdown("### 📥 Upload Your Document")

    uploaded_file = st.file_uploader(
        "Choose a file to analyze (PDF, PNG, JPG, JPEG)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload official PDF forms, admission circulars, scholarship guidelines, or clear photos."
    )

    # Determine file source (either direct upload or sample document preloaded)
    file_to_process = None
    file_name = None
    file_bytes = None

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        file_size_kb = len(file_bytes) / 1024
        file_type = uploaded_file.type or Path(file_name).suffix.upper()

        # File Metadata Display
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.metric("File Name", file_name)
        with meta_col2:
            st.metric("File Type", file_type)
        with meta_col3:
            st.metric("File Size", f"{file_size_kb:.1f} KB")

        # Analyze Button
        analyze_clicked = st.button("🚀 Analyze Document", type="primary", use_container_width=True)

        effective_key = get_groq_api_key()

        # Handle Live AI Analysis Execution
        if analyze_clicked:
            if not effective_key:
                st.error("⚠️ AI service is not configured. Please ensure GROQ_API_KEY is in your .env file.")
            else:
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Step 1: File received
                        status_text.markdown("⏳ **File received...**")
                        progress_bar.progress(15)
                        time.sleep(0.3)

                        # Step 2: Extracting text
                        status_text.markdown("📄 **Extracting text from document...**")
                        progress_bar.progress(35)
                        extracted_text = extract_document_text(file_name, file_bytes)
                        st.session_state.document_text = extracted_text
                        time.sleep(0.3)

                        # Step 3: Understanding document
                        status_text.markdown("🧠 **Understanding document with AI (Llama-3 via Groq)...**")
                        progress_bar.progress(60)
                        time.sleep(0.3)

                        # Step 4: Identifying requirements & JSON analysis
                        status_text.markdown("🔍 **Identifying requirements and confusing fields...**")
                        progress_bar.progress(85)
                        try:
                            result = analyze_document_with_ai(
                                document_text=extracted_text,
                                target_language=target_language,
                                api_key=effective_key
                            )
                        except TypeError:
                            result = analyze_document_with_ai(
                                document_text=extracted_text,
                                api_key=effective_key
                            )
                        time.sleep(0.2)

                        # Step 5: Machine Learning Evaluation (Authenticity & Urgency via Logistic Regression)
                        status_text.markdown("🤖 **Running Machine Learning Risk & Authenticity Analysis (Logistic Regression)...**")
                        progress_bar.progress(92)
                        ml_evaluation = assess_document_ml(extracted_text, result)
                        st.session_state.ml_assessment = ml_evaluation
                        time.sleep(0.2)

                        # Step 6: Finalizing action plan
                        status_text.markdown("✅ **Finalizing your personalized action plan...**")
                        progress_bar.progress(100)
                        time.sleep(0.2)

                        # Save to session state
                        st.session_state.analysis_result = result
                        st.session_state.document_name = file_name
                        st.session_state.is_demo_mode = False
                        st.session_state.checklist_state = {item: False for item in result.get("action_items", [])}
                        st.session_state.chat_messages = []
                        st.session_state.current_lang = target_language  # Track language at analysis time

                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()

                    except DocumentProcessingError as dpe:
                        status_text.empty()
                        progress_bar.empty()
                        st.error(f"❌ {str(dpe)}")
                    except ValueError as ve:
                        status_text.empty()
                        progress_bar.empty()
                        st.error(f"⚠️ {str(ve)}")
                    except Exception as e:
                        status_text.empty()
                        progress_bar.empty()
                        st.error(f"The AI service is temporarily unavailable. Details: {str(e)}")

# ==========================================
# RESULTS DASHBOARD
# ==========================================
if st.session_state.analysis_result:
    res = st.session_state.analysis_result

    # --- Language-Change Detection: Re-run AI analysis if language switched ---
    if st.session_state.get("current_lang") != target_language and st.session_state.get("document_text"):
        with st.spinner("Translating analysis to selected language..."):
            try:
                new_result = analyze_document_with_ai(
                    document_text=st.session_state.document_text,
                    target_language=target_language,
                    api_key=st.session_state.get("custom_groq_key") or os.getenv("GROQ_API_KEY", "")
                )
            except TypeError:
                new_result = analyze_document_with_ai(
                    document_text=st.session_state.document_text,
                    api_key=st.session_state.get("custom_groq_key") or os.getenv("GROQ_API_KEY", "")
                )
            st.session_state.analysis_result = new_result
            st.session_state.current_lang = target_language
            st.session_state.checklist_state = {item: False for item in new_result.get("action_items", [])}
            st.session_state.chat_messages = []   # Reset chat — language changed
            # ml_assessment stays as-is (ML is language-agnostic)
            st.rerun()

    st.markdown("---")

    # Header with title and action button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader("📊 Document Intelligence Dashboard")
    with header_col2:
        if st.button("🔄 Upload New Document", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.document_text = None
            st.session_state.document_name = None
            st.session_state.checklist_state = {}
            st.session_state.chat_messages = []
            st.session_state.ml_assessment = None
            st.rerun()

    # MACHINE LEARNING INTELLIGENCE: DUAL LOGISTIC REGRESSION MODELS
    if not st.session_state.get("ml_assessment") and st.session_state.get("document_text"):
        st.session_state.ml_assessment = assess_document_ml(st.session_state.document_text, res)

    ml_eval = st.session_state.get("ml_assessment")
    if ml_eval:
        auth_data = ml_eval.get("authenticity", {})
        urg_data = ml_eval.get("urgency", {})

        #st.markdown("#### 🤖 Machine Learning Intelligence *(Logistic Regression)*")
        ml_col1, ml_col2 = st.columns(2)

        with ml_col1:
            auth_score = auth_data.get("authenticity_score", 90.0)
            auth_color = auth_data.get("badge_color", "#10B981")
            verdict = auth_data.get("verdict", "Verified Notice")
            confidence = auth_data.get("confidence", "High Confidence")
            signals_html = "".join(f"<li style='margin-bottom: 2px;'>{sig}</li>" for sig in auth_data.get("signals", []))

            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1.5px solid {auth_color}45; border-top: 4px solid {auth_color}; border-radius: 12px; padding: 1.1rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <span style="font-weight: 700; font-size: 0.98rem; color: #1E293B;">🛡️ Notice Authenticity & Trust Score</span>
                    <span style="background: {auth_color}18; color: {auth_color}; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">{confidence}</span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.3rem;">
                    <span style="font-size: 1.7rem; font-weight: 800; color: {auth_color};">{auth_score}%</span>
                    <span style="font-size: 0.92rem; font-weight: 600; color: #334155;">{verdict}</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.5rem;">
                    Check: <strong>Statutory Pattern & Verification Audit</strong>
                </div>
                <div style="font-size: 0.85rem; color: #475569;">
                    <strong>Verification Findings:</strong>
                    <ul style="margin: 0.25rem 0 0 1.1rem; padding: 0;">
                        {signals_html}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ml_col2:
            urg_score = urg_data.get("urgency_score", 50.0)
            urg_color = urg_data.get("badge_color", "#F59E0B")
            urg_level = urg_data.get("urgency_level", "Moderate Urgency")
            days_left = urg_data.get("days_left")
            days_label = f"{days_left} days left" if days_left is not None else "Timeline estimated"
            risk_factors_html = "".join(f"<li style='margin-bottom: 2px;'>{rf}</li>" for rf in urg_data.get("risk_factors", []))

            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1.5px solid {urg_color}45; border-top: 4px solid {urg_color}; border-radius: 12px; padding: 1.1rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <span style="font-weight: 700; font-size: 0.98rem; color: #1E293B;">⏱️ Filing Urgency & Action Priority</span>
                    <span style="background: {urg_color}18; color: {urg_color}; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">{days_label}</span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.3rem;">
                    <span style="font-size: 1.7rem; font-weight: 800; color: {urg_color};">{urg_score}%</span>
                    <span style="font-size: 0.92rem; font-weight: 600; color: #334155;">{urg_level}</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.5rem;">
                    Check: <strong>Filing Timeline & Requirement Burden</strong>
                </div>
                <div style="font-size: 0.85rem; color: #475569;">
                    <strong>Risk Drivers:</strong>
                    <ul style="margin: 0.25rem 0 0 1.1rem; padding: 0;">
                        {risk_factors_html}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 1. HERO SECTION: WHAT YOU NEED TO DO (Action Checklist)
    action_items = res.get("action_items", [])
    if action_items:
        # Calculate completion rate
        total_actions = len(action_items)
        completed_actions = sum(1 for item in action_items if st.session_state.checklist_state.get(item, False))
        percent_done = int((completed_actions / total_actions) * 100) if total_actions > 0 else 0

        st.markdown("""
        <div class="hero-action-card">
            <div class="hero-action-title">
                <span>🎯</span> WHAT YOU NEED TO DO (YOUR ACTION PLAN)
            </div>
            <p style="color: #15803D; font-size: 0.95rem; margin-top: 0.2rem; margin-bottom: 0.8rem;">
                Check off each task as you complete it. Your progress is saved during this session.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(percent_done / 100)
        st.caption(f"**Progress:** {completed_actions} of {total_actions} tasks completed ({percent_done}%)")

        # Checklist Items
        for idx, item in enumerate(action_items):
            # Checkbox with persistent state
            current_checked = st.session_state.checklist_state.get(item, False)
            new_checked = st.checkbox(
                item,
                value=current_checked,
                key=f"check_item_{idx}"
            )
            if new_checked != current_checked:
                st.session_state.checklist_state[item] = new_checked
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. KEY HIGHLIGHTS: DEADLINE & REQUIRED DOCUMENTS
    col_left, col_right = st.columns([1, 2])

    with col_left:
        # IMPORTANT DEADLINE
        deadline_text = res.get("deadline", "Not specified")
        if not deadline_text or deadline_text.strip().lower() in ["not specified", "none", "unknown"]:
            disp_deadline = "Deadline not specified in the document"
            is_urgent = False
        else:
            disp_deadline = deadline_text
            is_urgent = True

        st.markdown(f"""
        <div class="deadline-card">
            <div class="deadline-title">⏰ Important Deadline</div>
            <div class="deadline-value" style="font-size: {'1.4rem' if not is_urgent else '1.75rem'};">
                {disp_deadline}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # DOCUMENT OVERVIEW
        with st.container():
            st.markdown("""<div class="info-card">""", unsafe_allow_html=True)
            st.markdown("#### 📄 Document Overview")
            st.markdown(f"**Title:** {res.get('document_title', 'Not specified')}")
            st.markdown(f"**Type:** `{res.get('document_type', 'Official Document')}`")
            st.markdown(f"**Summary:** {res.get('summary', 'No summary available.')}")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # REQUIRED DOCUMENTS
        req_docs = res.get("required_documents", [])
        with st.container():
            st.markdown("""<div class="info-card">""", unsafe_allow_html=True)
            st.markdown("#### 📑 Required Documents to Submit")
            if req_docs:
                for doc_item in req_docs:
                    st.markdown(f"- 🗂️ **{doc_item}**")
            else:
                st.info("No specific supporting documents are required according to this document.")
            st.markdown("</div>", unsafe_allow_html=True)

    # 3. CONFUSING FIELDS EXPLAINED
    important_fields = res.get("important_fields", [])
    if important_fields:
        st.markdown("### 💡 Confusing Fields & Requirements Explained")
        st.caption("We translate complex institutional jargon into simple explanations and show the exact certificate needed.")

        for f_idx, field_info in enumerate(important_fields):
            fname = field_info.get("field", "Field")
            fexp = field_info.get("explanation", "No explanation available.")
            fdoc = field_info.get("required_document", "None specified")

            doc_badge = f"<span class='field-doc'>📎 Supporting Document: {fdoc}</span>" if fdoc and fdoc.lower() not in ["none specified", "none", "not specified"] else ""

            st.markdown(f"""
            <div class="field-card">
                <div class="field-name">{fname}</div>
                <div class="field-desc"><b>What it means:</b> {fexp}</div>
                {doc_badge}
            </div>
            """, unsafe_allow_html=True)

    # 4. WARNINGS & PENALTIES
    warnings = res.get("warnings", [])
    if warnings and any(w.strip() for w in warnings):
        st.markdown("### ⚠️ Important Warnings & Restrictions")
        for warn in warnings:
            if warn.strip():
                st.markdown(f"""
                <div class="warning-box">
                    <b>Warning:</b> {warn}
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # ASK FORMFIX AI (Grounded Chat Interface)
    # ==========================================
    st.markdown("---")
    st.markdown("### 💬 Ask FormFix AI About This Document")
    st.caption("Ask questions strictly based on the uploaded document. The AI will never invent facts outside the text.")

    # Suggested Questions Chips
    st.markdown("**Suggested Questions:**")
    chip_col1, chip_col2, chip_col3, chip_col4, chip_col5 = st.columns(5)

    prompt_to_send = None

    with chip_col1:
        if st.button("⏰ What is the deadline?", key="chip_deadline", use_container_width=True):
            prompt_to_send = "What is the deadline?"
    with chip_col2:
        if st.button("📑 Which documents do I need?", key="chip_docs", use_container_width=True):
            prompt_to_send = "Which documents do I need?"
    with chip_col3:
        if st.button("💰 Do I need income certificate?", key="chip_income", use_container_width=True):
            prompt_to_send = "Do I need an income certificate?"
    with chip_col4:
        if st.button("🏁 What should I do first?", key="chip_first", use_container_width=True):
            prompt_to_send = "What should I do first?"
    with chip_col5:
        if st.button("📝 Explain in simple words", key="chip_simple", use_container_width=True):
            prompt_to_send = "Explain this document in simple words."

    # Display Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    user_query = st.chat_input("Ask any question about your uploaded document...")

    if user_query:
        prompt_to_send = user_query

    if prompt_to_send:
        # Add user message to history
        st.session_state.chat_messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Generate Grounded Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document for answer..."):
                doc_text = st.session_state.document_text or ""
                active_api_key = get_groq_api_key()

                if not active_api_key:
                    reply = "AI service is not configured. Please ensure GROQ_API_KEY is in your .env file."
                else:
                    try:
                        reply = ask_document_ai(
                            document_text=doc_text,
                            question=prompt_to_send,
                            chat_history=st.session_state.chat_messages,
                            target_language=st.session_state.get("target_language", "English"),
                            api_key=active_api_key
                        )
                    except TypeError:
                        reply = ask_document_ai(
                            document_text=doc_text,
                            question=prompt_to_send,
                            chat_history=st.session_state.chat_messages,
                            api_key=active_api_key
                        )

                st.markdown(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
