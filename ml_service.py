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
        """
        if not text or len(text.strip()) < 30:
            return {
                "is_authentic": True,
                "authenticity_score": 85.0,
                "fraud_risk_score": 15.0,
                "verdict": "Unverified (Document text too short for full statistical evaluation)",
                "confidence": "Low",
                "signals": ["Document contains minimal text; fallback analysis applied."],
                "badge_color": "#F59E0B"
            }

        # Predict probability of class 1 (Authentic)
        proba_authentic = float(self.pipeline.predict_proba([text])[0][1])
        auth_percent = round(proba_authentic * 100, 1)
        fraud_percent = round((1.0 - proba_authentic) * 100, 1)

        # Detect specific linguistic and structural cues
        text_lower = text.lower()
        signals: List[str] = []

        # Authentic cues
        if re.search(r'\b(ref|memo|no|notification)\s*[:./\-]?\s*[\w\-/]+', text_lower):
            if target_language == "Hindi":
                signals.append("आधिकारिक संदर्भ/विपणन संख्या प्रारूप पता चला")
            elif target_language == "Marathi":
                signals.append("अधिकृत संदर्भ/विपणन क्रमांक स्वरूप आढळले")
            else:
                signals.append("Official reference/dispatch number format detected")
        if re.search(r'\b(registrar|controller|director|secretary|competent authority|examinations)\b', text_lower):
            if target_language == "Hindi":
                signals.append("मानक प्रशासनिक पदनाम पता चला")
            elif target_language == "Marathi":
                signals.append("मानक प्रशासकीय पदनाम आढळले")
            else:
                signals.append("Standard administrative designation detected")
        if re.search(r'\b(gazette|resolution|act|stipulated|pursuant|ordinance|annexure)\b', text_lower):
            if target_language == "Hindi":
                signals.append("औपचारिक वैधानिक शब्दावली मेल")
            elif target_language == "Marathi":
                signals.append("औपचारिक वैधानिक शब्दसंग्रह जुळला")
            else:
                signals.append("Formal statutory vocabulary match")
        if re.search(r'\b(\.gov\.in|\.nic\.in|\.ac\.in|\.edu)\b', text_lower):
            if target_language == "Hindi":
                signals.append("आधिकारिक सरकारी/शैक्षणिक डोमेन संदर्भ मौजूद हैं")
            elif target_language == "Marathi":
                signals.append("अधिकृत सरकारी/शैक्षणिक डोमेन संदर्भ आहेत")
            else:
                signals.append("Official government/academic domain references present")

        # Fraud / Scam cues
        if re.search(r'\b(upi|gpay|phonepe|paytm|qr code|transfer fee|processing fee)\b', text_lower):
            if target_language == "Hindi":
                signals.append("⚠️ भुगतान अनुरोध (UPI/वॉलेट/प्रत्यक्ष शुल्क) पता चला")
            elif target_language == "Marathi":
                signals.append("⚠️ पैसे मागणी (UPI/वॉलेट/थेट फी) आढळली")
            else:
                signals.append("⚠️ Payment solicitation (UPI/Wallet/Direct Fee) detected")
        if re.search(r'\b(telegram|t\.me|whatsapp group|dm for|call on whatsapp)\b', text_lower):
            if target_language == "Hindi":
                signals.append("⚠️ अनौपचारिक सोशल चैनल रीडायरेक्शन (टेलीग्राम/व्हाट्सएप)")
            elif target_language == "Marathi":
                signals.append("⚠️ अनौपचारिक सामाजिक चैनल पुनर्निर्देशन (टेलिग्राम/व्हॉट्सअॅप)")
            else:
                signals.append("⚠️ Informal social channel redirection (Telegram/WhatsApp)")
        if re.search(r'\b(guaranteed selection|no exam direct|free laptop|lottery|prize|hurry up)\b', text_lower):
            if target_language == "Hindi":
                signals.append("⚠️ अवास्तविक वादा या घोटाला ट्रिगर वाक्यांश पहचाना गया")
            elif target_language == "Marathi":
                signals.append("⚠️ अवास्तव वचन किंवा फसवणुकीचे ट्रिगर वाक्यांश ओळखले")
            else:
                signals.append("⚠️ Unrealistic promise or scam trigger phrasing identified")
        if re.search(r'\b(otp|cvv|debit card pin|atm pin)\b', text_lower):
            if target_language == "Hindi":
                signals.append("🚨 उच्च-जोखिम क्रेडेंशियल हार्वेस्टिंग पैटर्न पता चला")
            elif target_language == "Marathi":
                signals.append("🚨 उच्च-धोका क्रेडेन्शियल हार्वेस्टिंग पॅटर्न आढळले")
            else:
                signals.append("🚨 High-risk credential harvesting pattern detected")

        # Heuristic calibration if strong fraud signals exist
        if any(s.startswith("⚠️") or s.startswith("🚨") for s in signals):
            auth_percent = min(auth_percent, 28.0)
            fraud_percent = max(fraud_percent, 72.0)

        is_authentic = auth_percent >= 50.0

        if is_authentic:
            if auth_percent >= 80:
                if target_language == "Hindi":
                    verdict = "सत्यापित आधिकारिक नोटिस"
                elif target_language == "Marathi":
                    verdict = "सत्यापित अधिकृत परिपत्रक"
                else:
                    verdict = "Verified Official Notice"
                badge_color = "#10B981"  # Emerald Green
                if target_language == "Hindi":
                    confidence = "उच्च विश्वास"
                elif target_language == "Marathi":
                    confidence = "उच्च विश्वासार्हता"
                else:
                    confidence = "High Confidence"
            else:
                if target_language == "Hindi":
                    verdict = "संभावित आधिकारिक परिपत्र"
                elif target_language == "Marathi":
                    verdict = "संभाव्य अधिकृत परिपत्रक"
                else:
                    verdict = "Likely Official Circular"
                badge_color = "#3B82F6"  # Blue
                if target_language == "Hindi":
                    confidence = "मध्यम विश्वास"
                elif target_language == "Marathi":
                    confidence = "मध्यम विश्वासार्हता"
                else:
                    confidence = "Moderate Confidence"
        else:
            if fraud_percent >= 70:
                if target_language == "Hindi":
                    verdict = "संदिग्ध / संभावित घोटाला नोटिस"
                elif target_language == "Marathi":
                    verdict = "संशयास्पद / संभाव्य फसवणुकीचे परिपत्रक"
                else:
                    verdict = "Suspicious / Potential Scam Notice"
                badge_color = "#EF4444"  # Red
                if target_language == "Hindi":
                    confidence = "उच्च संदेह"
                elif target_language == "Marathi":
                    confidence = "उच्च संशय"
                else:
                    confidence = "High Suspicion"
            else:
                if target_language == "Hindi":
                    verdict = "असत्यापित / अनियमित नोटिस"
                elif target_language == "Marathi":
                    verdict = "असत्यापित / अनियमित परिपत्रक"
                else:
                    verdict = "Unverified / Irregular Notice"
                badge_color = "#F59E0B"  # Amber
                if target_language == "Hindi":
                    confidence = "समीक्षा सलाह"
                elif target_language == "Marathi":
                    confidence = "समीक्षा सलाह"
                else:
                    confidence = "Review Advised"

        if not signals:
            if target_language == "Hindi":
                signals.append("मानक प्रशासनिक संचार पैटर्न का विश्लेषण किया गया")
            elif target_language == "Marathi":
                signals.append("मानक प्रशासकीय संचार नमुने विश्लेषित")
            else:
                signals.append("Standard administrative communication patterns analyzed")

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
        """
        risk_factors = []
        text_lower = full_text.lower()

        # Feature 1: Days remaining to deadline
        days_left = self._estimate_days_left(deadline_text)
        if days_left is not None:
            if days_left <= 3:
                if target_language == "Hindi":
                    risk_factors.append(f"तत्काल डेडलाइन: केवल {days_left} दिन शेष")
                elif target_language == "Marathi":
                    risk_factors.append(f"तातडीची अंतिम मुदत: फक्त {days_left} दिवस शिल्लक")
                else:
                    risk_factors.append(f"Immediate deadline: Only {days_left} day(s) remaining")
            elif days_left <= 7:
                if target_language == "Hindi":
                    risk_factors.append(f"आगामी डेडलाइन: {days_left} दिन शेष")
                elif target_language == "Marathi":
                    risk_factors.append(f"आगामी अंतिम मुदत: {days_left} दिवस शिल्लक")
                else:
                    risk_factors.append(f"Upcoming deadline: {days_left} days remaining")
        else:
            # Default fallback when deadline is unspecified: assume moderate 14 days
            days_left = 14
            if target_language == "Hindi":
                risk_factors.append("कोई स्पष्ट डेडलाइन नहीं मिली; मानक समयरेखा के साथ मूल्यांकन किया गया")
            elif target_language == "Marathi":
                risk_factors.append("कोणतीही स्पष्ट अंतिम मुदत आढळली नाही; मानक कालरेषेसह मूल्यांकन")
            else:
                risk_factors.append("No explicit deadline found; evaluated with standard timeline")

        # Feature 2: Required document count
        doc_count = len(required_docs) if required_docs else 0
        if doc_count >= 5:
            if target_language == "Hindi":
                risk_factors.append(f"भारी दस्तावेजीकरण बोझ: {doc_count} अनिवार्य प्रमाणपत्र आवश्यक")
            elif target_language == "Marathi":
                risk_factors.append(f"जड कागदपत्र भार: {doc_count} अनिवार्य प्रमाणपत्रे आवश्यक")
            else:
                risk_factors.append(f"Heavy documentation burden: {doc_count} mandatory certificates needed")
        elif doc_count >= 3:
            if target_language == "Hindi":
                risk_factors.append(f"मानक दस्तावेजीकरण: {doc_count} आवश्यक दस्तावेज")
            elif target_language == "Marathi":
                risk_factors.append(f"मानक कागदपत्रीकरण: {doc_count} आवश्यक कागदपत्रे")
            else:
                risk_factors.append(f"Standard documentation: {doc_count} required documents")

        # Feature 3: Penalty / Disqualification clauses
        has_penalty = 1.0 if re.search(
            r'\b(summarily rejected|late fee|forfeited|disqualified|no extension|invalidated|debarred|penalty)\b',
            text_lower
        ) else 0.0
        if has_penalty:
            if target_language == "Hindi":
                risk_factors.append("कठोर अयोग्यता या दंड खंड पता चला")
            elif target_language == "Marathi":
                risk_factors.append("कठोर अपात्रता किंवा दंड तरतुद आढळली")
            else:
                risk_factors.append("Strict disqualification or penalty clause detected")

        # Feature 4: Offline / Postal / Notary requirement
        is_offline = 1.0 if re.search(
            r'\b(speed post|registered post|hard copy|physical submission|in person|by hand|notary|stamp paper|affidavit)\b',
            text_lower
        ) else 0.0
        if is_offline:
            if target_language == "Hindi":
                risk_factors.append("भौतिक अनुपालन बाधा (स्पीड पोस्ट, भौतिक यात्रा, या नोटरी स्टांप आवश्यक)")
            elif target_language == "Marathi":
                risk_factors.append("भौतिक अनुपालन अडथळा (स्पीड पोस्ट, भौतिक भेट, किंवा नोटरी स्टॅम्प आवश्यक)")
            else:
                risk_factors.append("Physical compliance barrier (Speed post, physical visit, or notary stamp required)")

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
        """
        feat_vector, days_left, risk_factors = self.extract_features(
            deadline_text, required_docs, action_items, full_text, target_language
        )

        feat_scaled = self.scaler.transform(feat_vector)
        # Probability of High Urgency (class 1)
        proba_urgent = float(self.clf.predict_proba(feat_scaled)[0][1])
        urgency_score = round(proba_urgent * 100, 1)

        # Categorize
        if urgency_score >= 70.0 or (days_left is not None and days_left <= 3):
            if target_language == "Hindi":
                urgency_level = "महत्वपूर्ण तात्कालिकता (तत्काल कार्य)"
                recommendation = "उच्च प्राथमिकता: इस कटऑफ को याद करने से बचने के लिए आज से अनिवार्य दस्तावेज इकट्ठा करना शुरू करें।"
            elif target_language == "Marathi":
                urgency_level = "महत्त्वाची तातडी (तातडीची कृती)"
                recommendation = "उच्च प्राधान्य: या अंतिम मुदतीस गुमावण्यापासून वाचण्यासाठी आजपासून अनिवार्य कागदपत्रे गोळा करण्यास सुरुवात करा."
            else:
                urgency_level = "Critical Urgency (Immediate Action)"
                recommendation = "High priority: Start gathering mandatory documents today to avoid missing this cutoff."
            badge_color = "#DC2626"  # Red
        elif urgency_score >= 38.0:
            if target_language == "Hindi":
                urgency_level = "मध्यम तात्कालिकता"
                recommendation = "मानक समयरेखा: अगले कुछ दिनों के भीत प्रमाणपत्र संग्रह और ऑनलाइन भरने की योजना बनाएं।"
            elif target_language == "Marathi":
                urgency_level = "मध्यम तातडी"
                recommendation = "मानक कालरेखा: पुढील काही दिवसांत प्रमाणपत्रे गोळा करणे आणि ऑनलाइन भरण्याची योजना आखा."
            else:
                urgency_level = "Moderate Urgency"
                recommendation = "Standard timeline: Plan certificate collection and online filling within the next few days."
            badge_color = "#F59E0B"  # Amber
        else:
            if target_language == "Hindi":
                urgency_level = "नियमित / कम तात्कालिकता"
                recommendation = "आरामदायक खिड़की उपलब्ध है। नियमित तैयारी के साथ आगे बढ़ें।"
            elif target_language == "Marathi":
                urgency_level = "नियमित / कमी तातडी"
                recommendation = "आरामदायक कालावधी उपलब्ध आहे. नियमित तयारीसह पुढे जा."
            else:
                urgency_level = "Routine / Low Urgency"
                recommendation = "Comfortable window available. Proceed with regular preparation."
            badge_color = "#10B981"  # Green

        return {
            "urgency_score": urgency_score,
            "urgency_level": urgency_level,
            "badge_color": badge_color,
            "days_left": days_left,
            "risk_factors": risk_factors,
            "recommendation": recommendation
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
    Returns authenticity prediction and urgency risk assessment.
    """
    auth_model, urg_model = get_ml_models()

    deadline_text = analysis_result.get("deadline", "")
    req_docs = analysis_result.get("required_documents", [])
    action_items = analysis_result.get("action_items", [])

    auth_pred = auth_model.predict(document_text, target_language)
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
