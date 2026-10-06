"""
FormFix AI - Machine Learning Service
Dual Logistic Regression Models:
1. Notice Authenticity & Fraud/Spam Detector (NLP: TF-IDF + Logistic Regression)
2. Deadline Urgency & Action Risk Engine (Tabular Feature Engineering + Logistic Regression)
"""

import re
import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==============================================================================
# MULTILINGUAL TRANSLATION TABLE FOR ALL ML OUTPUT STRINGS
# ==============================================================================

_ML_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ── Authenticity signals ─────────────────────────────────────────────────
    "Official reference/dispatch number format detected": {
        "Hindi":   "आधिकारिक संदर्भ/डिस्पैच नंबर प्रारूप पाया गया",
        "Marathi": "अधिकृत संदर्भ/डिस्पॅच क्रमांक स्वरूप आढळले"
    },
    "Standard administrative designation detected": {
        "Hindi":   "मानक प्रशासनिक पदनाम पाया गया",
        "Marathi": "मानक प्रशासकीय पदनाम आढळले"
    },
    "Formal statutory vocabulary match": {
        "Hindi":   "औपचारिक कानूनी शब्दावली का मिलान हुआ",
        "Marathi": "औपचारिक वैधानिक शब्दसंग्रह जुळला"
    },
    "Official government/academic domain references present": {
        "Hindi":   "आधिकारिक सरकारी/शैक्षणिक डोमेन संदर्भ मौजूद हैं",
        "Marathi": "अधिकृत शासकीय/शैक्षणिक डोमेन संदर्भ आढळले"
    },
    "Standard administrative communication patterns analyzed": {
        "Hindi":   "मानक प्रशासनिक संचार पैटर्न का विश्लेषण किया गया",
        "Marathi": "मानक प्रशासकीय संवाद नमुन्यांचे विश्लेषण केले"
    },
    "Document contains minimal text; fallback analysis applied.": {
        "Hindi":   "दस्तावेज़ में बहुत कम पाठ है; वैकल्पिक विश्लेषण लागू किया गया।",
        "Marathi": "दस्तऐवजात अत्यंत कमी मजकूर आहे; पर्यायी विश्लेषण लागू केले."
    },
    # ── Fraud signals ────────────────────────────────────────────────────────
    "⚠️ Payment solicitation (UPI/Wallet/Direct Fee) detected": {
        "Hindi":   "⚠️ UPI/वॉलेट/डायरेक्ट शुल्क मांगा गया — यह संदिग्ध है",
        "Marathi": "⚠️ UPI/वॉलेट/डायरेक्ट शुल्क मागणी आढळली — हे संशयास्पद आहे"
    },
    "⚠️ Informal social channel redirection (Telegram/WhatsApp)": {
        "Hindi":   "⚠️ अनौपचारिक सोशल चैनल (Telegram/WhatsApp) पर भेजने की कोशिश",
        "Marathi": "⚠️ अनौपचारिक सामाजिक वाहिनी (Telegram/WhatsApp) वर वळवण्याचा प्रयत्न"
    },
    "⚠️ Unrealistic promise or scam trigger phrasing identified": {
        "Hindi":   "⚠️ अवास्तविक वादा या धोखाधड़ी को बढ़ावा देने वाली भाषा पहचानी गई",
        "Marathi": "⚠️ अवास्तव वचन किंवा फसवणूक प्रवृत्त करणारी भाषा ओळखली"
    },
    "🚨 High-risk credential harvesting pattern detected": {
        "Hindi":   "🚨 उच्च जोखिम — बैंकिंग/व्यक्तिगत जानकारी चुराने का पैटर्न पाया गया",
        "Marathi": "🚨 उच्च धोका — बँकिंग/वैयक्तिक माहिती चोरण्याचा नमुना आढळला"
    },
    # ── Verdict labels ───────────────────────────────────────────────────────
    "Verified Official Notice": {
        "Hindi": "सत्यापित आधिकारिक नोटिस", "Marathi": "सत्यापित अधिकृत परिपत्रक"
    },
    "Likely Official Circular": {
        "Hindi": "संभवतः आधिकारिक परिपत्र", "Marathi": "बहुधा अधिकृत परिपत्रक"
    },
    "Suspicious / Potential Scam Notice": {
        "Hindi": "संदिग्ध / संभावित धोखाधड़ी नोटिस", "Marathi": "संशयास्पद / संभाव्य फसवणूक नोटीस"
    },
    "Unverified / Irregular Notice": {
        "Hindi": "असत्यापित / अनियमित नोटिस", "Marathi": "असत्यापित / अनियमित नोटीस"
    },
    "Unverified (Document text too short for full statistical evaluation)": {
        "Hindi":   "असत्यापित (पूर्ण सांख्यिकीय मूल्यांकन के लिए दस्तावेज़ बहुत छोटा है)",
        "Marathi": "असत्यापित (पूर्ण सांख्यिकीय मूल्यांकनासाठी दस्तऐवज खूप लहान आहे)"
    },
    # ── Confidence labels ────────────────────────────────────────────────────
    "High Confidence": {
        "Hindi": "उच्च विश्वास", "Marathi": "उच्च विश्वासार्हता"
    },
    "Moderate Confidence": {
        "Hindi": "मध्यम विश्वास", "Marathi": "मध्यम विश्वासार्हता"
    },
    "High Suspicion": {
        "Hindi": "अत्यधिक संदिग्ध", "Marathi": "अत्यंत संशयास्पद"
    },
    "Review Advised": {
        "Hindi": "समीक्षा की सिफारिश की जाती है", "Marathi": "पुनरावलोकन करण्याचा सल्ला"
    },
    "Low": {
        "Hindi": "कम", "Marathi": "कमी"
    },
    # ── Urgency level labels ─────────────────────────────────────────────────
    "Critical Urgency (Immediate Action)": {
        "Hindi": "अत्यावश्यक (तुरंत कार्रवाई करें)", "Marathi": "अत्यंत तातडीचे (त्वरित कारवाई करा)"
    },
    "Moderate Urgency": {
        "Hindi": "मध्यम तात्कालिकता", "Marathi": "मध्यम तातडी"
    },
    "Routine / Low Urgency": {
        "Hindi": "सामान्य / कम तात्कालिकता", "Marathi": "सामान्य / कमी तातडी"
    },
    # ── Risk factor messages ─────────────────────────────────────────────────
    "No explicit deadline found; evaluated with standard timeline": {
        "Hindi":   "कोई स्पष्ट समय सीमा नहीं मिली; मानक समयरेखा के साथ मूल्यांकन किया गया",
        "Marathi": "कोणतीही स्पष्ट अंतिम मुदत आढळली नाही; मानक कालरेषेनुसार मूल्यांकन केले"
    },
    "Strict disqualification or penalty clause detected": {
        "Hindi":   "सख्त अयोग्यता या जुर्माना खंड पाया गया",
        "Marathi": "कडक अपात्रता किंवा दंड कलम आढळले"
    },
    "Physical compliance barrier (Speed post, physical visit, or notary stamp required)": {
        "Hindi":   "भौतिक अनुपालन बाधा (स्पीड पोस्ट, व्यक्तिगत उपस्थिति, या नोटरी स्टाम्प आवश्यक)",
        "Marathi": "भौतिक अनुपालन अडथळा (स्पीड पोस्ट, प्रत्यक्ष भेट, किंवा नोटरी शिक्का आवश्यक)"
    },
}


def _t(text: str, lang: str) -> str:
    """Translate an ML output string to the target language. Falls back to English."""
    if lang == "English":
        return text
    table = _ML_TRANSLATIONS.get(text, {})
    return table.get(lang, text)


def _t_days(days: int, lang: str) -> str:
    """Translate risk factor strings that contain dynamic day counts."""
    if lang == "English":
        if days <= 3:
            return f"Immediate deadline: Only {days} day(s) remaining"
        return f"Upcoming deadline: {days} days remaining"
    elif lang == "Hindi":
        if days <= 3:
            return f"तत्काल समय सीमा: केवल {days} दिन शेष हैं"
        return f"आगामी समय सीमा: {days} दिन शेष हैं"
    else:  # Marathi
        if days <= 3:
            return f"तातडीची अंतिम मुदत: फक्त {days} दिवस उरले आहेत"
        return f"आगामी अंतिम मुदत: {days} दिवस शिल्लक आहेत"


def _t_docs(count: int, lang: str) -> str:
    """Translate documentation burden risk factor with dynamic count."""
    if lang == "English":
        if count >= 5:
            return f"Heavy documentation burden: {count} mandatory certificates needed"
        return f"Standard documentation: {count} required documents"
    elif lang == "Hindi":
        if count >= 5:
            return f"अधिक दस्तावेज़ी बोझ: {count} अनिवार्य प्रमाणपत्र आवश्यक हैं"
        return f"मानक दस्तावेज़ीकरण: {count} आवश्यक दस्तावेज़"
    else:  # Marathi
        if count >= 5:
            return f"अधिक कागदपत्रांचा भार: {count} अनिवार्य प्रमाणपत्रे आवश्यक"
        return f"मानक कागदपत्रे: {count} आवश्यक दस्तऐवज"




# ==============================================================================
# 1. NLP MODEL: NOTICE AUTHENTICITY & FRAUD/SPAM DETECTOR
# ==============================================================================

# Curated benchmark dataset of official gazettes/circulars vs fake/scam notices
_AUTHENTIC_SAMPLES = [
    "GOVERNMENT OF INDIA MINISTRY OF EDUCATION Notification Ref No: F.1-4/2026-UGC(CPP-II) dated 12th August 2026. In exercise of powers conferred under section 12 of the University Grants Commission Act.",
    "DEPARTMENT OF HIGHER EDUCATION Circular Memo No: DHE/SCH/2026/894. All eligible undergraduate and postgraduate scholars are hereby informed to submit application on the state scholarship portal.",
    "CONTROLLER OF EXAMINATIONS State Technical University Ref: STU/COE/EXAM/2026/1042. Notification regarding schedule of semester examinations November 2026. Hall tickets available on official university domain.",
    "DIRECTORATE OF TECHNICAL EDUCATION Official Gazette Order. All polytechnic and engineering colleges must verify applicant credentials, domicile certificates, and caste validity in accordance with Government Resolution.",
    "STAFF SELECTION COMMISSION Notice No: SSC/CGL/2026/ADM. Online applications are invited from eligible citizens of India. Candidates are advised to read the detailed brochure on ssc.nic.in before registering.",
    "OFFICE OF THE REGISTRAR National Institute of Technology. Circular: Academic Session 2026-27 hostel allotment and mess fee deposit through official SBI Collect portal only.",
    "STATE COMMON ENTRANCE TEST CELL Information Brochure for Admission to First Year Undergraduate Technical Courses. Submission of online application and physical document verification at Facilitation Centers.",
    "EMPLOYEES PROVIDENT FUND ORGANISATION Circular No. EPFO/COMP/2026/771. Guidelines regarding timely submission of joint option declarations through member portal.",
    "UNIVERSITY GRANTS COMMISSION Public Notice regarding recognition of Open and Distance Learning programs for academic year 2026-2027. Refer official portal ugc.ac.in for approved university lists.",
    "MINISTRY OF SOCIAL JUSTICE AND EMPOWERMENT Guidelines for National Fellowship. Disbursement directly to Aadhaar-seeded bank accounts through PFMS DBT mechanism."
]

_FRAUD_SCAM_SAMPLES = [
    "URGENT NOTICE!! Free laptops and ₹25,000 scholarship for all 10th and 12th students under PM New Scheme! Click this link to register and send processing fee of ₹499 via UPI/GPay to secure slot!",
    "Direct Job Recruitment in Railway Board 2026! No exam, direct selection for 12,500 clerk posts. Pay registration charge of ₹999 to WhatsApp number 9876543210 immediately. Limited seats available hurry!",
    "EXAM POSTPONED!! All semester exams scheduled next week are cancelled due to emergency order. Join our Telegram channel t.me/exam_leaks to get updated leaked question papers and bypass verification.",
    "Congratulations! You have been selected for State Merit Cash Award. Transfer ₹350 verification fee to Google Pay UPI id govt-award@upi within 2 hours or your prize will be cancelled permanently!",
    "Work from home govt data entry project. Earn ₹50,000 monthly. Deposit ₹1,500 refundable security deposit to personal Paytm account to receive appointment letter. Call now on WhatsApp.",
    "URGENT Circular: Special grace marks bonus of 15% declared by University. Pay ₹1,200 evaluation charges to direct UPI QR code before midnight to update your marksheet instantly.",
    "Govt subsidy loan scheme approved! 80% subsidy with zero interest. Send Aadhaar card photo, bank OTP, and ₹2,000 file charge to private email govt-loan-scheme@gmail.com for instant transfer.",
    "Guaranteed selection in upcoming police recruitment. Only 10 seats left. DM on Telegram @recruitment_head for secret selection list. ₹5,000 advance required via PhonePe.",
    "Notice: Student scholarship fund release. Your payment is on hold. Click bit.ly/claim-scholarship-cash and enter debit card PIN and CVV to claim ₹15,000 scholarship right now.",
    "Official Alert!! Final warning: Your exam registration will be blocked today. Transfer ₹750 late clearance penalty directly to UPI merchant id immediately to unblock."
]


class NoticeAuthenticityModel:
    """
    Logistic Regression classifier with TF-IDF vectorization to detect
    official government/university circulars vs. fraudulent/scam notices.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=800,
                stop_words='english',
                sublinear_tf=True
            )),
            ('clf', LogisticRegression(
                C=1.5,
                max_iter=300,
                random_state=42
            ))
        ])
        self._train()

    def _train(self):
        X = _AUTHENTIC_SAMPLES + _FRAUD_SCAM_SAMPLES
        # 1 = Authentic, 0 = Fraud/Scam
        y = [1] * len(_AUTHENTIC_SAMPLES) + [0] * len(_FRAUD_SCAM_SAMPLES)
        self.pipeline.fit(X, y)

    def predict(self, text: str, target_language: str = "English") -> Dict[str, Any]:
        """
        Predict authenticity of a document using Logistic Regression.
        All output strings are returned in the requested target_language.
        """
        if not text or len(text.strip()) < 30:
            return {
                "is_authentic": True,
                "authenticity_score": 85.0,
                "fraud_risk_score": 15.0,
                "verdict": _t("Unverified (Document text too short for full statistical evaluation)", target_language),
                "confidence": _t("Low", target_language),
                "signals": [_t("Document contains minimal text; fallback analysis applied.", target_language)],
                "badge_color": "#F59E0B"
            }

        # Predict probability of class 1 (Authentic)
        proba_authentic = float(self.pipeline.predict_proba([text])[0][1])
        auth_percent = round(proba_authentic * 100, 1)
        fraud_percent = round((1.0 - proba_authentic) * 100, 1)

        # Detect specific linguistic and structural cues (always in English for regex matching)
        text_lower = text.lower()
        signals_en: List[str] = []

        # Authentic cues
        if re.search(r'\b(ref|memo|no|notification)\s*[:./\-]?\s*[\w\-/]+', text_lower):
            signals_en.append("Official reference/dispatch number format detected")
        if re.search(r'\b(registrar|controller|director|secretary|competent authority|examinations)\b', text_lower):
            signals_en.append("Standard administrative designation detected")
        if re.search(r'\b(gazette|resolution|act|stipulated|pursuant|ordinance|annexure)\b', text_lower):
            signals_en.append("Formal statutory vocabulary match")
        if re.search(r'\b(\.gov\.in|\.nic\.in|\.ac\.in|\.edu)\b', text_lower):
            signals_en.append("Official government/academic domain references present")

        # Fraud / Scam cues
        if re.search(r'\b(upi|gpay|phonepe|paytm|qr code|transfer fee|processing fee)\b', text_lower):
            signals_en.append("⚠️ Payment solicitation (UPI/Wallet/Direct Fee) detected")
        if re.search(r'\b(telegram|t\.me|whatsapp group|dm for|call on whatsapp)\b', text_lower):
            signals_en.append("⚠️ Informal social channel redirection (Telegram/WhatsApp)")
        if re.search(r'\b(guaranteed selection|no exam direct|free laptop|lottery|prize|hurry up)\b', text_lower):
            signals_en.append("⚠️ Unrealistic promise or scam trigger phrasing identified")
        if re.search(r'\b(otp|cvv|debit card pin|atm pin)\b', text_lower):
            signals_en.append("🚨 High-risk credential harvesting pattern detected")

        # Heuristic calibration if strong fraud signals exist
        if any(s.startswith("⚠️") or s.startswith("🚨") for s in signals_en):
            auth_percent = min(auth_percent, 28.0)
            fraud_percent = max(fraud_percent, 72.0)

        is_authentic = auth_percent >= 50.0

        if is_authentic:
            if auth_percent >= 80:
                verdict_en = "Verified Official Notice"
                badge_color = "#10B981"  # Emerald Green
                confidence_en = "High Confidence"
            else:
                verdict_en = "Likely Official Circular"
                badge_color = "#3B82F6"  # Blue
                confidence_en = "Moderate Confidence"
        else:
            if fraud_percent >= 70:
                verdict_en = "Suspicious / Potential Scam Notice"
                badge_color = "#EF4444"  # Red
                confidence_en = "High Suspicion"
            else:
                verdict_en = "Unverified / Irregular Notice"
                badge_color = "#F59E0B"  # Amber
                confidence_en = "Review Advised"

        if not signals_en:
            signals_en.append("Standard administrative communication patterns analyzed")

        # Translate all output strings to target language
        signals = [_t(s, target_language) for s in signals_en]
        verdict = _t(verdict_en, target_language)
        confidence = _t(confidence_en, target_language)

        return {
            "is_authentic": is_authentic,
            "authenticity_score": auth_percent,
            "fraud_risk_score": fraud_percent,
            "verdict": verdict,
            "confidence": confidence,
            "signals": signals,
            "badge_color": badge_color
        }


# ==============================================================================
# 2. TABULAR MODEL: DEADLINE URGENCY & ACTION RISK ENGINE
# ==============================================================================

# Engineered training dataset for administrative urgency prediction
# Feature Columns:
# [days_remaining, num_required_docs, has_penalty_clause (0/1), is_offline_submission (0/1), action_items_count]
# Label: 1 = High Urgency / Critical Risk, 0 = Normal / Low Urgency
_URGENCY_TRAINING_X = np.array([
    # High Urgency scenarios (Tight timeline, heavy requirements, penalties, offline)
    [1.0, 5.0, 1.0, 1.0, 8.0],
    [2.0, 4.0, 1.0, 0.0, 6.0],
    [3.0, 6.0, 1.0, 1.0, 7.0],
    [2.0, 2.0, 1.0, 0.0, 4.0],
    [4.0, 5.0, 1.0, 1.0, 8.0],
    [5.0, 7.0, 1.0, 1.0, 9.0],
    [3.0, 3.0, 0.0, 1.0, 5.0],
    [1.0, 1.0, 1.0, 0.0, 3.0],
    [4.0, 4.0, 1.0, 0.0, 6.0],
    [6.0, 8.0, 1.0, 1.0, 10.0],
    [5.0, 5.0, 1.0, 0.0, 7.0],
    [2.0, 7.0, 0.0, 1.0, 8.0],

    # Moderate / Low Urgency scenarios (Ample time, digital submissions, fewer requirements)
    [14.0, 2.0, 0.0, 0.0, 3.0],
    [21.0, 3.0, 0.0, 0.0, 4.0],
    [30.0, 4.0, 1.0, 0.0, 5.0],
    [25.0, 2.0, 0.0, 0.0, 3.0],
    [45.0, 3.0, 0.0, 0.0, 4.0],
    [18.0, 1.0, 0.0, 0.0, 2.0],
    [35.0, 5.0, 0.0, 1.0, 6.0],
    [28.0, 3.0, 1.0, 0.0, 4.0],
    [60.0, 4.0, 0.0, 0.0, 5.0],
    [20.0, 2.0, 0.0, 0.0, 3.0],
    [15.0, 3.0, 0.0, 0.0, 4.0],
    [40.0, 1.0, 0.0, 0.0, 2.0],
])

_URGENCY_TRAINING_Y = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # High Urgency
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # Low / Moderate Urgency
])


class UrgencyRiskModel:
    """
    Logistic Regression classifier trained on engineered administrative risk features
    to forecast urgency and submission risk levels.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.2, random_state=42)
        self._train()

    def _train(self):
        X_scaled = self.scaler.fit_transform(_URGENCY_TRAINING_X)
        self.clf.fit(X_scaled, _URGENCY_TRAINING_Y)

    def extract_features(
        self,
        deadline_text: str,
        required_docs: List[str],
        action_items: List[str],
        full_text: str,
        target_language: str = "English"
    ) -> Tuple[np.ndarray, Optional[int], List[str]]:
        """
        Transform raw document analysis into quantitative feature vector.
        Risk factor strings are returned in target_language.
        """
        risk_factors = []
        text_lower = full_text.lower()

        # Feature 1: Days remaining to deadline
        days_left = self._estimate_days_left(deadline_text)
        if days_left is not None:
            if days_left <= 7:
                risk_factors.append(_t_days(days_left, target_language))
        else:
            # Default fallback when deadline is unspecified: assume moderate 14 days
            days_left = 14
            risk_factors.append(_t("No explicit deadline found; evaluated with standard timeline", target_language))

        # Feature 2: Required document count
        doc_count = len(required_docs) if required_docs else 0
        if doc_count >= 3:
            risk_factors.append(_t_docs(doc_count, target_language))

        # Feature 3: Penalty / Disqualification clauses
        has_penalty = 1.0 if re.search(
            r'\b(summarily rejected|late fee|forfeited|disqualified|no extension|invalidated|debarred|penalty)\b',
            text_lower
        ) else 0.0
        if has_penalty:
            risk_factors.append(_t("Strict disqualification or penalty clause detected", target_language))

        # Feature 4: Offline / Postal / Notary requirement
        is_offline = 1.0 if re.search(
            r'\b(speed post|registered post|hard copy|physical submission|in person|by hand|notary|stamp paper|affidavit)\b',
            text_lower
        ) else 0.0
        if is_offline:
            risk_factors.append(_t("Physical compliance barrier (Speed post, physical visit, or notary stamp required)", target_language))

        # Feature 5: Action items count
        action_count = len(action_items) if action_items else 4

        feature_vector = np.array([[float(days_left), float(doc_count), has_penalty, is_offline, float(action_count)]])
        return feature_vector, days_left, risk_factors

    def predict(
        self,
        deadline_text: str,
        required_docs: List[str],
        action_items: List[str],
        full_text: str,
        target_language: str = "English"
    ) -> Dict[str, Any]:
        """
        Calculate urgency score and classification using Logistic Regression.
        All output strings are in target_language.
        """
        feat_vector, days_left, risk_factors = self.extract_features(
            deadline_text, required_docs, action_items, full_text, target_language
        )

        feat_scaled = self.scaler.transform(feat_vector)
        # Probability of High Urgency (class 1)
        proba_urgent = float(self.clf.predict_proba(feat_scaled)[0][1])
        urgency_score = round(proba_urgent * 100, 1)

        # Categorize and translate urgency level
        if urgency_score >= 70.0 or (days_left is not None and days_left <= 3):
            urgency_level = _t("Critical Urgency (Immediate Action)", target_language)
            badge_color = "#DC2626"  # Red
        elif urgency_score >= 38.0:
            urgency_level = _t("Moderate Urgency", target_language)
            badge_color = "#F59E0B"  # Amber
        else:
            urgency_level = _t("Routine / Low Urgency", target_language)
            badge_color = "#10B981"  # Green

        return {
            "urgency_score": urgency_score,
            "urgency_level": urgency_level,
            "badge_color": badge_color,
            "days_left": days_left,
            "risk_factors": risk_factors,
        }

    @staticmethod
    def _estimate_days_left(deadline_text: str) -> Optional[int]:
        """Extract date from deadline text and compute days difference from today."""
        if not deadline_text or deadline_text.strip().lower() in ["not specified", "none", "unknown", ""]:
            return None

        # Try to parse standard dates like "15 October 2026", "15/10/2026", "2026-10-15"
        date_patterns = [
            r'(\d{1,2})[\s\-]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+(\d{4})',
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
            r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})'
        ]

        today = datetime.date.today()

        for pattern in date_patterns:
            match = re.search(pattern, deadline_text, re.IGNORECASE)
            if match:
                try:
                    matched_str = match.group(0)
                    for fmt in ("%d %B %Y", "%d %b %Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
                        try:
                            parsed_date = datetime.datetime.strptime(matched_str, fmt).date()
                            delta = (parsed_date - today).days
                            return max(0, delta)
                        except ValueError:
                            continue
                except Exception:
                    pass

        # Check for relative keywords like "today", "tomorrow", "3 days"
        text_lower = deadline_text.lower()
        if "today" in text_lower:
            return 0
        if "tomorrow" in text_lower:
            return 1
        day_match = re.search(r'(\d+)\s+days?', text_lower)
        if day_match:
            return int(day_match.group(1))

        return None


# ==============================================================================
# GLOBAL SINGLETONS & CONVENIENCE API
# ==============================================================================

_authenticity_model = None
_urgency_model = None

def get_ml_models() -> Tuple[NoticeAuthenticityModel, UrgencyRiskModel]:
    """Lazy initialize ML models to keep startup snappy."""
    global _authenticity_model, _urgency_model
    if _authenticity_model is None:
        _authenticity_model = NoticeAuthenticityModel()
    if _urgency_model is None:
        _urgency_model = UrgencyRiskModel()
    return _authenticity_model, _urgency_model


def assess_document_ml(
    document_text: str,
    analysis_result: Dict[str, Any],
    target_language: str = "English"
) -> Dict[str, Any]:
    """
    Run full dual-model Machine Learning assessment on a document.
    Returns authenticity prediction and urgency risk assessment,
    with all output strings in the requested target_language.
    """
    auth_model, urg_model = get_ml_models()

    deadline_text = analysis_result.get("deadline", "")
    req_docs = analysis_result.get("required_documents", [])
    action_items = analysis_result.get("action_items", [])

    auth_pred = auth_model.predict(document_text, target_language=target_language)
    urg_pred = urg_model.predict(
        deadline_text=deadline_text,
        required_docs=req_docs,
        action_items=action_items,
        full_text=document_text,
        target_language=target_language
    )

    return {
        "authenticity": auth_pred,
        "urgency": urg_pred
    }
