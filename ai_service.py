import os
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
FAST_GROQ_MODEL = "qwen/qwen3.8-27b"
MODELS_TO_TRY = [
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b"
]

SAMPLE_SCHOLARSHIP_DATA = {
    "document_title": "State Merit-Cum-Means Scholarship Renewal Notice (2026-2027)",
    "document_type": "Scholarship Renewal Notice",
    "summary": "Official notification from the Department of Higher Education inviting eligible undergraduate and postgraduate scholars to apply for scholarship renewal for the 2026-2027 academic session.",
    "deadline": "15 October 2026",
    "required_documents": [
        "Previous semester marksheets (minimum 60% aggregate)",
        "Bonafide Student Certificate issued by Dean/Principal",
        "Competent Authority Income Certificate (valid for current financial year)",
        "Latest College Fee Receipt with transaction ID",
        "Bank Passbook copy linked to student Aadhaar number"
    ],
    "important_fields": [
        {
            "field": "Annual Family Income",
            "explanation": "Total gross income of all earning family members combined from all sources during the financial year. Must not exceed Rs. 2,50,000/- per annum.",
            "required_document": "Income Certificate"
        },
        {
            "field": "Bonafide Certificate",
            "explanation": "Official declaration issued by the Head of the Institution certifying that the applicant is an active full-time student.",
            "required_document": "Bonafide Student Certificate"
        },
        {
            "field": "Previous Academic Aggregate",
            "explanation": "Cumulative percentage or SGPA/CGPA obtained in the preceding two semesters. Minimum required is 60% with no active backlogs.",
            "required_document": "Previous Semester Marksheets"
        },
        {
            "field": "Fee Receipt Number",
            "explanation": "Official institutional acknowledgment of tuition fee payment for the current academic year.",
            "required_document": "College Fee Receipt"
        }
    ],
    "action_items": [
        "Collect previous semester marksheets verifying minimum 60% aggregate",
        "Obtain signed Bonafide Student Certificate from Principal/Dean's office",
        "Verify Annual Family Income is under Rs. 2,50,000 and get Income Certificate",
        "Download latest college fee payment receipt with transaction ID",
        "Confirm bank account is active and seeded with Aadhaar",
        "Fill online renewal form on the state scholarship portal",
        "Upload scanned self-attested copies of all mandatory certificates",
        "Submit completed application online before 15 October 2026 (5:00 PM IST)"
    ],
    "warnings": [
        "Incomplete applications or applications with illegible attachments will be rejected without notice.",
        "Submission after the strict deadline of 15 October 2026 will not be accepted under any circumstances.",
        "Any discrepancy or false declaration regarding family income will result in immediate cancellation and recovery of funds.",
        "Students having active backlogs or repeat papers are not eligible for renewal."
    ]
}

def get_groq_api_key() -> Optional[str]:
    """Retrieve the Groq API key from environment, .env file, or Streamlit Cloud secrets."""
    # 1. Check Streamlit Cloud secrets (when deployed to share.streamlit.io)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            secret_key = st.secrets["GROQ_API_KEY"]
            if secret_key and secret_key.strip():
                return secret_key.strip()
    except Exception:
        pass

    # 2. Check local .env file
    try:
        from pathlib import Path
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=True)
        else:
            load_dotenv(override=True)
    except Exception:
        pass

    # 3. Check environment variable
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip() and key.strip() != "your_api_key_here":
        return key.strip()

    # 4. Fallback: check if .env has a raw key line starting with gsk_
    try:
        from pathlib import Path
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                clean_line = line.strip().replace('"', '').replace("'", "")
                if clean_line.startswith("gsk_"):
                    return clean_line
    except Exception:
        pass

    return None

def is_groq_configured() -> bool:
    """Check if a valid Groq API key is present."""
    return get_groq_api_key() is not None

def clean_json_response(raw_text: str) -> str:
    """Strip markdown code fence blocks if returned by the LLM."""
    raw_text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        return match.group(1).strip()
    return raw_text

def analyze_document_with_ai(
    document_text: str,
    target_language: str = "English",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sends document text to Groq LLM and extracts structured JSON analysis in the chosen language.
    """
    key = api_key or get_groq_api_key()
    if not key:
        raise ValueError("AI API key is not configured. Add GROQ_API_KEY to your .env file.")

    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("The 'groq' package is not installed. Please run: pip install groq")

    client = Groq(api_key=key)

    lang_instruction = ""
    if target_language and target_language.lower() != "english":
        lang_instruction = (
            f"TARGET LANGUAGE REQUIREMENT: {target_language.upper()}.\n"
            f"Regardless of the language of the uploaded document, you MUST write all values for "
            f"'summary', 'action_items', 'important_fields' (field names, explanations, and required documents), "
            f"'deadline', and 'warnings' in fluent, natural, grammatically correct {target_language} "
            f"(using proper Devanagari script for Hindi and Marathi). Do not translate JSON key names.\n\n"
        )

    system_prompt = (
        "You are an expert AI Document Action Assistant.\n"
        f"{lang_instruction}"
        "Your mission is: Document -> Understanding -> Action.\n"
        "You analyze official forms, notices, admission letters, scholarship applications, and government documents.\n"
        "Your job is to extract useful, crystal-clear, and actionable information from the supplied document.\n\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Do NOT invent or hallucinate information. If information is not present in the document, explicitly say 'Not specified' or use an empty list [].\n"
        "2. Do NOT behave like a generic summarizer. Focus intensely on WHAT THE USER NEEDS TO DO, DEADLINES, REQUIRED DOCUMENTS, and CONFUSING FIELDS.\n"
        "3. For important_fields, explain confusing terminology in simple everyday language, and indicate the exact supporting document required ONLY if mentioned or directly inferable from the document.\n"
        "4. For action_items: MUST strictly list positive, constructive chronological steps the user must take to prepare, collect documents, or apply. Do NOT put warnings, prohibitions, or things not to do in action_items.\n"
        "5. For warnings: MUST provide PROTECTIVE, RISK-BASED ADVISORIES. Do NOT merely repeat scam threats, fees, or punitive demands from the document as if they are legitimate rules. Instead, actively analyze risks and advise the user:\n"
        "   - If money/fees are demanded via UPI, GPay, PhonePe, QR code, or personal transfers: Explicitly WARN that this looks suspicious, noting that official government bodies do NOT collect fees through personal UPI or WhatsApp, and advise the user NOT to transfer money without verifying on an official .gov.in or university portal.\n"
        "   - If sensitive credentials (debit card, CVV, OTP, PIN) or informal social channels (Telegram/WhatsApp groups) are requested: Alert the user to high fraud/phishing risk and urge them never to share card or banking details.\n"
        "   - If strict penalties, short deadlines, or difficult attestations exist: Warn about the practical disqualification risk (e.g., 'Disqualification Risk: 2-day turnaround for physical postal submission may lead to late rejection').\n"
        "   - Always frame warnings from the applicant's safety perspective (e.g. '⚠️ Suspicious Requirement: Do not transfer money via UPI without official confirmation').\n\n"
        "You MUST respond ONLY with valid JSON matching this schema:\n"
        "{\n"
        '  "document_title": "Document title or Not specified",\n'
        '  "document_type": "e.g. Scholarship Notice, College Admission Form, Government Circular, etc.",\n'
        '  "summary": "Concise 2-3 sentence overview of what this document is about",\n'
        '  "deadline": "Specific deadline date and time, or Not specified",\n'
        '  "required_documents": ["Document 1", "Document 2"],\n'
        '  "important_fields": [\n'
        '    {\n'
        '      "field": "Field or term name",\n'
        '      "explanation": "Clear simple explanation of what it means",\n'
        '      "required_document": "Supporting document required, or None specified"\n'
        '    }\n'
        '  ],\n'
        '  "action_items": ["Step 1", "Step 2"],\n'
        '  "warnings": ["Warning 1", "Warning 2"]\n'
        "}"
    )

    user_prompt = f"Please analyze this document text and return the structured JSON:\n\n--- DOCUMENT TEXT ---\n{document_text}\n--- END DOCUMENT TEXT ---"

    models_to_try = MODELS_TO_TRY
    last_exception = None

    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            cleaned_content = clean_json_response(raw_content)
            data = json.loads(cleaned_content)
            
            # Validate core keys
            required_keys = ["document_title", "summary", "deadline", "required_documents", "important_fields", "action_items"]
            for rk in required_keys:
                if rk not in data:
                    data[rk] = "Not specified" if "documents" not in rk and "items" not in rk and "fields" not in rk else []

            return data
        except Exception as e:
            last_exception = e
            continue

    raise RuntimeError(f"The AI service is temporarily unavailable. Please check your API configuration or network connection. ({str(last_exception)})")

def ask_document_ai(
    document_text: str,
    question: str,
    chat_history: list = None,
    target_language: str = "English",
    api_key: Optional[str] = None
) -> str:
    """
    Answers user questions strictly grounded in the uploaded document in the selected language.
    """
    key = api_key or get_groq_api_key()
    if not key:
        return "AI API key is not configured. Add GROQ_API_KEY to your .env file."

    try:
        from groq import Groq
    except ImportError:
        return "The 'groq' package is not installed."

    client = Groq(api_key=key)

    lang_rule = ""
    if target_language and target_language.lower() != "english":
        lang_rule = f"\n5. Answer fluently and naturally in {target_language} (using proper Devanagari script for Hindi or Marathi)."

    system_prompt = (
        "You are FormFix AI's interactive document assistant.\n"
        "Your task is to answer the user's question based ONLY on the provided document text.\n"
        "STRICT RULES:\n"
        "1. Ground your answer completely in the document text provided.\n"
        "2. If the answer cannot be found in the document, state clearly in the target language that the information is not present in the document.\n"
        "3. Do not invent, guess, or bring in external facts.\n"
        "4. Keep your answer direct, clear, and actionable."
        f"{lang_rule}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"DOCUMENT CONTENT:\n{document_text}\n\nUSER QUESTION: {question}"
    })

    try:
        response = client.chat.completions.create(
            model=FAST_GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI service error: {str(e)}"

def get_demo_analysis() -> Dict[str, Any]:
    """Returns pre-verified sample analysis data for demo mode without calling external API."""
    return SAMPLE_SCHOLARSHIP_DATA.copy()
