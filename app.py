import streamlit as st

# --- SENIOR LOCALIZATION INITIALIZATION ---
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

def set_design():
    st.markdown("""
        <style>
        /* GLOBAL ACCESSIBILITY OVERRIDE */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBaseButton-secondary"] {
            background-color: #FFFFFF !important;
            font-family: 'Inter', sans-serif !important;
            color: #000000 !important;
        }

        /* Target all common text elements specifically */
        p, li, span, label, h1, h2, h3, h4, i, .stMarkdown, [data-testid="stMarkdownContainer"] {
            color: #000000 !important;
            font-family: 'Inter', sans-serif !important;
            background-color: transparent !important;
        }

        /* Bio-Card Refinement: Fix overlapping and z-index */
        .bio-card {
            background-color: #F8F9FA !important;
            padding: 2.5rem;
            border-radius: 15px;
            border: 2px solid #E0E0E0;
            margin-bottom: 2rem;
            position: relative;
            z-index: 10; /* Ensure card is above any ghost elements */
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            overflow: visible;
        }

        /* Hide overlapping chat labels ("smart assistant" labels) */
        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] + div > div:first-child {
            display: none !important; /* Hides the "assistant" or "user" label text */
        }

        /* Widget Visibility Aggression */
        [data-testid="stWidgetLabel"] p, .stSelectbox label, .stMultiSelect label, .stNumberInput label {
            color: #000000 !important;
            font-size: 1.15rem !important;
            font-weight: 750 !important;
            margin-bottom: 0.5rem !important;
        }

        /* Ensure input text is visible but not overlapping */
        input, select, .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] {
            color: #000000 !important;
            font-weight: 600 !important;
            background-color: #FFFFFF !important;
            border-radius: 8px !important;
        }

        /* Sidebar Visibility */
        [data-testid="stSidebar"] * {
            color: #000000 !important;
        }

        /* Chat Message Visibility */
        [data-testid="stChatMessage"] {
            background-color: #E8EAED !important;
            border: 1px solid #D1D5DB !important;
            margin-bottom: 12px !important;
        }
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span {
            color: #000000 !important;
            font-weight: 500 !important;
        }

        /* Button Polish */
        .stButton>button {
            background-color: #007BFF !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            border: none !important;
        }
        .stButton>button p {
            color: #FFFFFF !important;
        }

        /* Success/Info States */
        .stSuccess, .stInfo {
            background-color: #D1E7DD !important;
            color: #0F5132 !important;
            border: 2px solid #000000 !important;
            font-weight: 700 !important;
        }
        </style>
    """, unsafe_allow_html=True)

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import streamlit.components.v1 as components
import os
import re
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
from io import BytesIO
from db_manager import save_profile, log_health_check, get_last_checkin
from ocr_engine import extract_biomarkers, get_health_score_boost
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

# --- SPECIALIST & ANATOMICAL MAPPINGS ---
def get_specialist(disease):
    mapping = {
        "Migraine": "Neurologist",
        "Diabetes": "Endocrinologist",
        "Hypertension": "Cardiologist",
        "Glaucoma": "Ophthalmologist",
        "Common Cold": "General Physician",
        "Asthma": "Pulmonologist",
        "Bronchial Asthma": "Pulmonologist",
        "Fungal infection": "Dermatologist"
    }
    return mapping.get(disease, "General Physician")

def generate_pdf(bio_data, symptoms, results, lang_code, shap_summary, h_part):
    pdf = FPDF()
    pdf.add_page()
    
    # Fonts - Noto Sans for Indian script support (Placeholder for TTF loading)
    # To fully support Tamil/Hindi, the user must place 'NotoSans-Regular.ttf' in the root
    try:
        # pdf.add_font("NotoSans", "", "NotoSans-Regular.ttf", uni=True)
        # pdf.set_font("NotoSans", size=12)
        pdf.set_font("Arial", 'B', 16)
    except:
        pdf.set_font("Arial", 'B', 16)
    
    # Header
    pdf.set_text_color(7, 94, 84) # WhatsApp-style Green
    header_text = translate_dynamic("BIOPREDICT AI: UNIVERSAL DIAGNOSTIC SUMMARY", lang_code)
    pdf.cell(200, 10, txt=header_text, ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0)
    sub_text = translate_dynamic("Universal Healthcare Access Powered by BioPredict AI", lang_code)
    pdf.cell(200, 10, txt=sub_text, ln=True, align='C')
    pdf.ln(10)
    
    # 1. Patient Input (Symptoms)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="[1] Patient Input & Symptoms", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 8, txt=f"Age: {bio_data.get('age', 'N/A')} | Gender: {bio_data.get('gender', 'N/A')}", ln=True)
    pdf.multi_cell(0, 8, txt=f"Symptoms Identified: {', '.join(symptoms)}")
    pdf.ln(5)
    
    # 2. AI Diagnosis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="[2] AI Diagnostic Prediction", ln=True)
    pdf.set_font("Arial", size=11)
    top_d, conf = results[0]
    pdf.set_text_color(20, 184, 166)
    pdf.cell(200, 8, txt=f"Preliminary Diagnosis: {top_d} ({conf*100:.1f}% Confidence)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # 3. Explainability (XAI)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="[3] Explainability (XAI) Insights", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, txt=f"Primary clinical indicators: {shap_summary}")
    pdf.ln(5)

    # 4. Anatomical Context
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="[4] Anatomical Context", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 8, txt=f"3D Model Interaction: Highlighted {h_part} region.", ln=True)
    pdf.ln(5)
    
    # 5. Clinical Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="[5] Clinical Recommendations", ln=True)
    pdf.set_font("Arial", size=10)
    specialist = get_specialist(top_d)
    pdf.cell(200, 8, txt=f"- Specialist to Consult: {specialist}", ln=True)
    pdf.cell(200, 8, txt="- Monitor vitals every 4-6 hours.", ln=True)
    pdf.cell(200, 8, txt="- Maintain hydration and rest.", ln=True)
    
    # 6. Medical Disclaimer
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 8)
    disclaimer = "This is an AI-generated assessment for educational purposes. Consult a certified medical professional for formal diagnosis."
    pdf.multi_cell(0, 5, txt=f"DISCLAIMER: {disclaimer}")
    
    # Footer
    pdf.set_y(-25)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(0, 10, 'Developed by: S.S. Keerthi Vasan & M. Harishvasan', 0, 1, 'C')
    pdf.cell(0, 5, 'Reg No: 192519092', 0, 0, 'C')
    
    return bytes(pdf.output())

def get_shap_summary(vec, top_disease):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(np.array([vec]))
    class_idx = list(le.classes_).index(top_disease)
    # Get top 3 features
    vals = shap_vals[class_idx][0] if isinstance(shap_vals, list) else shap_vals[0, :, class_idx]
    top_indices = np.argsort(vals)[-3:][::-1]
    return ", ".join([symptoms_list[i] for i in top_indices if vals[i] > 0])

# --- PAGE CONFIG ---
st.set_page_config(page_title="BioAI - Hyper-Localized Health Companion", layout="wide", initial_sidebar_state="expanded")
set_design()

# --- HYPER-LOCALIZATION CONFIG (30+ LANGUAGES) ---
LANG_MAP = {
    "English": "en", "Hindi (हिन्दी)": "hi", "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr", "Bengali (বাংলা)": "bn", "Gujarati (ગુજરાતી)": "gu",
    "Kannada (ಕನ್ನಡ)": "kn", "Malayalam (മലയാളം)": "ml", "Punjabi (ਪੰਜਾਬী)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or", "Assamese (অসমীয়া)": "as", "Maithili (मैथिली)": "mai",
    "Santali (संताली)": "sat", "Kashmiri (کٲشُر)": "ks", "Konkani (कोंकणी)": "kok",
    "Sindhi (سنڌي)": "sd", "Dogri (डोंगरी)": "doi", "Manipuri (মণিপুরী)": "mni",
    "Sanskrit (संस्कृतम्)": "sa", "Nepali (नेपाली)": "ne", "Urdu (اردو)": "ur",
    "Bhojpuri (भोजपुरी)": "bho", "Haryanvi (हरियाणवी)": "bgc", "Rajasthani (राजस्थانی)": "raj",
    "Bodo (बड़ो)": "brx", "Mizo (मिज़ो)": "lus", "Khasi (खासी)": "kha", "Garo (गारो)": "grt",
    "Tulu (ತುಳು)": "tcy"
}

LANG_STRINGS = {
    "en": {
        "app_title": "🛡️ BioAi Health Companion",
        "app_subtitle": "Your Empathetic AI Medical Guide",
        "guardian_menu": "Guardian Menu",
        "localization_label": "Hyper-Localization",
        "symptom_label": "How are you feeling today?",
        "predict_btn": "Analyze Health Score",
        "bio_header": "Patient Medical Profile",
        "age": "Age", "gender": "Gender", "history": "Medical History",
        "init_btn": "Initialize Companion",
        "result_header": "Diagnostic Insights",
        "jargon_btn": "Translate to Simple Terms",
        "speak_btn": "Listen to Diagnosis",
        "next_steps_header": "Next Steps Checklist",
        "lab_report_header": "📂 Lab Report Analysis",
        "camera_label": "📸 Wound Vision Scan",
        "chat_placeholder": "Describe your condition...",
        "accuracy_meter": "AI Confidence Score",
        "download_report": "📄 Download Clinical Report",
        "new_consult_btn": "Start New Consultation",
        "male": "Male", "female": "Female", "other": "Other",
        "yes": "Yes", "no": "No",
        "checklist_item_1": "1. 💧 Increase hydration (2.5L water/day).",
        "checklist_item_2_head": "2. 💆 Relax in a low-light environment.",
        "checklist_item_3": "3. 🏥 If symptoms worsen, schedule an appointment immediately.",
        "probing_stage": "Probing Stage",
        "probing_refine": "I'm refining my analysis. Current Confidence in",
        "probing_question": "To be more precise, could you tell me: **Are you also experiencing",
        "final_analysis_msg": "Thank you for the details. I have finalized my analysis.",
        "final_summary_btn": "View Final Diagnostic Summary",
        "guidance_msg": "I've outlined your guidance below. You can also download a formal clinical report for your records.",
        "setup_msg": "Hello! I am your Health Companion. Let's set up your profile first.",
        "stop_btn": "Stop"
    },
    "hi": {
        "app_title": "🛡️ बायोएआई स्वास्थ्य साथी",
        "app_subtitle": "आपका सहानुभूतिपूर्ण एआई चिकित्सा गाइड",
        "guardian_menu": "गार्जियन मेनू",
        "localization_label": "हाइपर-लोकलाइजेशन",
        "symptom_label": "आज आप कैसा महसूस कर रहे हैं?",
        "predict_btn": "स्वास्थ्य स्कोर का विश्लेषण करें",
        "bio_header": "रोगी चिकित्सा प्रोफ़ाइल",
        "age": "आयु", "gender": "लिंग", "history": "चिकित्सा इतिहास",
        "init_btn": "साथी शुरू करें",
        "result_header": "नैदानिक अंतर्दற्ष्टि",
        "jargon_btn": "सरल शब्दों में अनुवाद करें",
        "speak_btn": "निदान सुनें",
        "next_steps_header": "अगले कदम चेकलिस्ट",
        "lab_report_header": "📂 लैब रिपोर्ट विश्लेषण",
        "camera_label": "📸 घाव दृष्टि स्कैन",
        "chat_placeholder": "अपनी स्थिति का वर्णन करें...",
        "accuracy_meter": "एआई विश्वास स्कोर",
        "download_report": "📄 क्लिनिकल रिपोर्ट डाउनलोड करें",
        "new_consult_btn": "नया परामर्श शुरू करें",
        "male": "पुरुष", "female": "महिला", "other": "अन्य",
        "yes": "हाँ", "no": "नहीं",
        "checklist_item_1": "1. 💧 जलयोजन बढ़ाएं (2.5 लीटर पानी/दिन)।",
        "checklist_item_2_head": "2. 💆 कम रोशनी वाले वातावरण में आराम करें।",
        "checklist_item_3": "3. 🏥 यदि लक्षण बिगड़ते हैं, तो तुरंत अपॉइंटमेंट लें।",
        "probing_stage": "जाँच चरण",
        "probing_refine": "मैं अपने विश्लेषण को परिष्कृत कर रहा हूँ। वर्तमान विश्वास",
        "probing_question": "अधिक सटीक होने के लिए, क्या आप मुझे बता सकते हैं: **क्या आप भी अनुभव कर रहे हैं",
        "final_analysis_msg": "विवरण के लिए धन्यवाद। मैंने अपना विश्लेषण अंतिम रूप दे दिया है।",
        "final_summary_btn": "अंतिम नैदानिक सारांश देखें",
        "guidance_msg": "मैंने नीचे आपके मार्गदर्शन की रूपरेखा दी है। आप अपने रिकॉर्ड के लिए एक औपचारिक नैदानिक रिपोर्ट भी डाउनलोड कर सकते हैं।",
        "setup_msg": "नमस्ते! मैं आपका स्वास्थ्य साथी हूँ। आइए पहले आपकी प्रोफ़ाइल सेट करें।",
        "stop_btn": "रोकें"
    },
    "ta": {
        "app_title": "🛡️ பயோஏஐ சுகாதாரத் துணை",
        "app_subtitle": "உங்கள் அனுதாபமுள்ள AI மருத்துவ வழிகாட்டி",
        "guardian_menu": "கார்டியன் மெனு",
        "localization_label": "ஹைப்பர்-லோக்கலைசேஷன்",
        "symptom_label": "இன்று நீங்கள் எப்படி உணருகிறீர்கள்?",
        "predict_btn": "சுகாதார மதிப்பெண்ணைப் பகுப்பாய்வு செய்யுங்கள்",
        "bio_header": "நோயாளி மருத்துவ விவரக்குறிப்பு",
        "age": "வயது", "gender": "பாலினம்", "history": "மருத்துவ வரலாறு",
        "init_btn": "துணையைத் தொடங்குங்கள்",
        "result_header": "கண்டறியும் நுண்ணறிவு",
        "jargon_btn": "எளிய சொற்களில் மொழிபெயர்க்கவும்",
        "speak_btn": "கண்டறிதலைக் கேளுங்கள்",
        "next_steps_header": "அடுத்த படிகள் சரிபார்ப்புப் பட்டியல்",
        "lab_report_header": "📂 ஆய்வக அறிக்கை பகுப்பாய்வு",
        "camera_label": "📸 காயம் பார்வை ஸ்கேன்",
        "chat_placeholder": "உங்கள் நிலையை விளக்குங்கள்...",
        "accuracy_meter": "AI நம்பிக்கை மதிப்பெண்",
        "download_report": "📄 மருத்துவ அறிக்கையைப் பதிவிறக்கவும்",
        "new_consult_btn": "புதிய ஆலோசனையைத் தொடங்குங்கள்",
        "male": "ஆண்", "female": "பெண்", "other": "மற்றவை",
        "yes": "ஆம்", "no": "இல்லை",
        "checklist_item_1": "1. 💧 நீரேற்றத்தை அதிகரிக்கவும் (ஒரு நாளைக்கு 2.5 லிட்டர் தண்ணீர்).",
        "checklist_item_2_head": "2. 💆 குறைந்த வெளிச்சம் உள்ள சூழலில் ஓய்வெடுக்கவும்.",
        "checklist_item_3": "3. 🏥 அறிகுறிகள் மோசமடைந்தால், உடனடியாக சந்திப்பைத் திட்டமிடுங்கள்.",
        "probing_stage": "ஆராய்ச்சி நிலை",
        "probing_refine": "நான் எனது பகுப்பாய்வைச் சீரமைக்கிறேன். தற்போதைய நம்பிக்கை",
        "probing_question": "இன்னும் துல்லியமாக இருக்க, நீங்கள் எனக்குச் சொல்ல முடியுமா: **நீங்களும் அனுபவிக்கிறீர்களா",
        "final_analysis_msg": "விவரங்களுக்கு நன்றி. எனது பகுப்பாய்வை நான் இறுதி செய்துவிட்டேன்.",
        "final_summary_btn": "இறுதி கண்டறியும் சுருக்கத்தைக் காண்க",
        "guidance_msg": "உங்கள் வழிகாட்டலை நான் கீழே கோடிட்டுக் காட்டியுள்ளேன். உங்கள் பதிவுகளுக்காக முறையான மருத்துவ அறிக்கையையும் நீங்கள் பதிவிறக்கம் செய்யலாம்.",
        "setup_msg": "வணக்கம்! நான் உங்கள் சுகாதாரத் துணை. முதலில் உங்கள் சுயவிவரத்தை அமைப்போம்.",
        "stop_btn": "நிறுத்து"
    }
}

def on_lang_change():
    st.session_state['lang'] = LANG_MAP[st.session_state.selected_lang_name]

def t(key):
    lang = st.session_state.get('lang', 'en')
    if lang in LANG_STRINGS and key in LANG_STRINGS[lang]: 
        return LANG_STRINGS[lang][key]
    
    # Fallback to English, then dynamic translation
    eng_text = LANG_STRINGS["en"].get(key, key)
    if lang == "en": return eng_text
    return translate_dynamic(eng_text, lang)

@st.cache_data
def symptom_format(symptom_name, lang):
    if lang == "en": return symptom_name
    return translate_dynamic(symptom_name, lang)

@st.cache_data
def translate_dynamic(text, dest_lang):
    if dest_lang == "en": return text
    try: return GoogleTranslator(source='auto', target=dest_lang).translate(text)
    except: return text

def speak_text(text, lang):
    try:
        # slow=True for medical clarity as requested
        tts = gTTS(text=text, lang=lang, slow=True)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except: return None

# --- STT & RECORDING ---
def record_audio(audio_bytes, lang_code):
    """Processes recorded audio bytes into text using Google Speech Recognition."""
    if not audio_bytes: return None
    r = sr.Recognizer()
    try:
        audio_stream = BytesIO(audio_bytes)
        with sr.AudioFile(audio_stream) as source:
            # Handle noise for cleaner transcription
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        # Recognize using Google API (supports 30+ Indian languages)
        text = r.recognize_google(audio, language=lang_code)
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# --- UI STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; font-size: 1.05rem; }
    .stApp { background: #F0F2F5; }
    .hero { background: linear-gradient(135deg, #075E54 0%, #128C7E 100%); color: white; padding: 40px; border-radius: 0 0 30px 30px; margin-bottom: 30px; text-align: center; }
    .hero h1 { font-size: 2.8rem; font-weight: 600; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.2rem; opacity: 0.9; }
    .prediction-card { background: white; padding: 25px; border-radius: 20px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stButton>button { border-radius: 10px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- MODELS ---
@st.cache_resource
def load_assets():
    with open('models/model.pkl', 'rb') as f: model = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    with open('models/symptoms_list.pkl', 'rb') as f: slist = pickle.load(f)
    return model, le, slist

model, le, symptoms_list = load_assets()

# --- STATE ---
if "step" not in st.session_state: st.session_state.step = 0
if "profile_id" not in st.session_state: st.session_state.profile_id = None
if "selected_symptoms" not in st.session_state: st.session_state.selected_symptoms = set()
if "biomarkers" not in st.session_state: st.session_state.biomarkers = {}
if "vision_data" not in st.session_state: st.session_state.vision_data = None
if "chat_log" not in st.session_state: st.session_state.chat_log = []
if "probing_count" not in st.session_state: st.session_state.probing_count = 0
if "probing_questions" not in st.session_state: st.session_state.probing_questions = []

# --- SIDEBAR ---
with st.sidebar:
    st.header(t('guardian_menu'))
    st.selectbox(t('localization_label'), 
                 options=list(LANG_MAP.keys()), 
                 key="selected_lang_name",
                 on_change=on_lang_change)
    
    st.markdown("---")
    st.header(t('lab_report_header'))
    lab_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if lab_file:
        markers = extract_biomarkers(lab_file.read())
        if markers:
            st.session_state.biomarkers = markers
            st.success(f"Extracted: {', '.join(markers.keys())}")
    
    st.markdown("---")
    st.header(t('camera_label'))
    vision_img = st.camera_input("")
    if vision_img:
        st.session_state.vision_data = "Localized Rash Detected"
        st.info("Vision engine mapping to avatar...")

# --- FLOW ---
# (Hero moved to main interface section)

def get_followup_question(current_symptoms, top_diseases):
    """Suggests a symptom to ask about based on high-probability diseases."""
    candidate_symptoms = []
    # Peek into model classes to find relevant symptoms for top diseases
    # Simplified: pick symptoms from symptoms_list that are common in top_diseases but not yet selected
    # In a real model, we'd use feature importance or correlation matrices
    # For now, we'll pick some common medical probes
    probes = ["Headache", "Fever", "Fatigue", "Nausea", "Dizziness", "Cough", "Breathlessness", "Skin Rash"]
    for p in probes:
        if p.lower() not in [s.lower() for s in current_symptoms]:
            candidate_symptoms.append(p)
    
    if not candidate_symptoms: return None
    # Pick a random one for variety or just the first
    return candidate_symptoms[0]

def t_jargon(disease):
    # Mock Jargon Translator
    jargon_map = {
        "Migraine": "A very bad headache that often causes nausea and sensitivity to light.",
        "Diabetes": "When your body has too much sugar in the blood.",
        "Hypertension": "When the force of blood against your artery walls is too high.",
        "Glaucoma": "An eye condition that can damage vision if too much pressure builds up."
    }
    desc = jargon_map.get(disease, "A complex condition affecting your health system.")
    return translate_dynamic(desc, st.session_state.lang)

def unified_inference(symptoms, bio_data, biomarkers):
    vec = [1 if s in symptoms else 0 for s in symptoms_list]
    probs = model.predict_proba([vec])[0]
    classes = list(le.classes_)
    
    # Weights: Medical History
    if bio_data.get("history"):
        for h in bio_data["history"]:
            if h in classes: probs[classes.index(h)] *= 1.4
    
    # Weights: Biomarkers (OCR)
    boosts = get_health_score_boost(biomarkers)
    for disease, boost in boosts.items():
        if disease in classes: probs[classes.index(disease)] *= boost
        
    probs = probs / np.sum(probs)
    top_indices = np.argsort(probs)[-3:][::-1]
    return list(zip(le.inverse_transform(top_indices), probs[top_indices])), vec

# (Utility functions moved below)

# --- UI HELPERS ---
def render_voice_input():
    if st.session_state.step == 1:
        c1, c2 = st.columns([8, 2])
        with c1:
            clinical_text = st.chat_input(t('chat_placeholder'))
        with c2:
            st.write("🎙️")
            voice_recording = mic_recorder(start_prompt=t('speak_btn'), stop_prompt=t('stop_btn'), key='recorder')
            if voice_recording:
                # Use the new record_audio function
                transcribed = record_audio(voice_recording['bytes'], st.session_state.lang)
                if transcribed:
                    clinical_text = transcribed
        return clinical_text
    return None

def render_3d_avatar(h_part, is_speaking=False):
    three_js = f"""
    <div id="scene" style="width:100%; height:300px; background:white; border-radius:20px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, 1.5, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias:true, alpha:true}});
        renderer.setSize(450, 300);
        document.getElementById('scene').appendChild(renderer.domElement);
        camera.position.z = 6;
        const mat = new THREE.MeshPhongMaterial({{color: 0x128C7E}});
        const human = new THREE.Group(); scene.add(human);
        human.add(new THREE.Mesh(new THREE.BoxGeometry(1.2, 2.5, 0.6), mat));
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.45), mat); head.position.y = 1.6; human.add(head);
        if("{h_part}" === "Head") head.material = new THREE.MeshPhongMaterial({{color: 0x075E54, emissive: 0x128C7E}});
        const l = new THREE.DirectionalLight(0xffffff, 1); l.position.set(5,5,5); scene.add(l);
        scene.add(new THREE.AmbientLight(0x404040, 0.8));
        (function anim() {{ 
            requestAnimationFrame(anim); 
            human.rotation.y += 0.005; 
            if({str(is_speaking).lower()}) {{
                human.scale.set(1 + Math.sin(Date.now() * 0.01) * 0.05, 1 + Math.sin(Date.now() * 0.01) * 0.05, 1);
            }}
            renderer.render(scene, camera); 
        }})();
    </script>
    """
    components.html(three_js, height=320)

# --- MAIN INTERFACE ---
st.markdown('<div class="bio-card" style="text-align: center;">', unsafe_allow_html=True)
st.markdown(f"""
    <div class="hero">
        <h1>{t('app_title')}</h1>
        <p>{t('app_subtitle')}</p>
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- CHAT FLOW ---
for msg in st.session_state.chat_log:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if st.session_state.step == 0:
    st.markdown('<div class="bio-card">', unsafe_allow_html=True)
    with st.chat_message("assistant"):
        st.write(t('setup_msg'))
        st.header(t('bio_header'))
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(t('age'), 0, 120, 25)
            gender = st.selectbox(t('gender'), [t('male'), t('female'), t('other')])
        with col2:
            # history = st.multiselect(t('history'), options=list(le.classes_))
            history = st.multiselect(t('history'), options=list(le.classes_), format_func=lambda x: symptom_format(x, st.session_state.lang))
        
        if st.button(t('init_btn')):
            st.session_state.profile_id = save_profile(age, gender, ",".join(history))
            # Translate the initialization message
            init_content = translate_dynamic(f"Profile initialized. I noticed you mentioned {', '.join(history) if history else 'no previous history'}. I'm here to help.", st.session_state.lang)
            st.session_state.chat_log.append({"role": "assistant", "content": init_content})
            st.session_state.step = 1
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.step == 1:
    # Check memory
    last_log = get_last_checkin(st.session_state.profile_id)
    if last_log and not st.session_state.chat_log:
        welcome_back = translate_dynamic(f"Welcome back. During our last talk, we discussed {last_log[1]}. How are you feeling today?", st.session_state.lang)
        st.session_state.chat_log.append({"role": "assistant", "content": welcome_back})
    
    with st.chat_message("assistant"): st.write(t('symptom_label'))
    
    desc = render_voice_input()
    if desc:
        st.session_state.chat_log.append({"role": "user", "content": desc})
        extracted = [s for s in symptoms_list if re.search(r'\b' + re.escape(s) + r'\b', desc.lower())]
        st.session_state.selected_symptoms.update(extracted)
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    results, vec = unified_inference(st.session_state.selected_symptoms, st.session_state.get("bio_data", {}), st.session_state.biomarkers)
    top_d = results[0][0]
    trans_top = translate_dynamic(top_d, st.session_state.lang)
    
    # Probing Phase
    if st.session_state.probing_count < 3: # 3 probes for a solid 2-4 range
        q_symptom = get_followup_question(st.session_state.selected_symptoms, [r[0] for r in results[:2]])
        # Translate the symptom itself for the question
        trans_q = translate_dynamic(q_symptom, st.session_state.lang)
        
        with st.chat_message("assistant"):
            st.write(f"### 🧪 {t('probing_stage')} {st.session_state.probing_count + 1}/3")
            st.write(f"{t('probing_refine')} {trans_top}: **{results[0][1]*100:.1f}%**")
            st.progress(results[0][1])
            st.write(f"{t('probing_question')} {trans_q}?**")
            
            # Voice output for the question
            q_speech = f"{t('probing_question')} {trans_q}?"
            audio_data = speak_text(q_speech, st.session_state.lang)
            if audio_data: st.audio(audio_data, format="audio/mp3")

            col1, col2 = st.columns(2)
            if col1.button(t('yes'), key=f"yes_{st.session_state.probing_count}"):
                st.session_state.selected_symptoms.add(q_symptom.lower())
                st.session_state.probing_count += 1
                st.session_state.chat_log.append({"role": "user", "content": f"{t('yes')}, I have {trans_q}."})
                st.rerun()
            if col2.button(t('no'), key=f"no_{st.session_state.probing_count}"):
                st.session_state.probing_count += 1
                st.session_state.chat_log.append({"role": "user", "content": f"{t('no')}, I don't have {trans_q}."})
                st.rerun()
    else:
        with st.chat_message("assistant"):
            st.write(f"{t('final_analysis_msg')} (Final Confidence: {results[0][1]*100:.1f}%)")
            if st.button(t('final_summary_btn')):
                st.session_state.step = 3
                st.rerun()


elif st.session_state.step == 3:
    st.markdown('<div class="bio-card">', unsafe_allow_html=True)
    results, vec = unified_inference(st.session_state.selected_symptoms, st.session_state.get("bio_data", {}), st.session_state.biomarkers)
    top_d = results[0][0]
    trans_top = translate_dynamic(top_d, st.session_state.lang)
    shap_summary = get_shap_summary(vec, top_d)
    
    with st.chat_message("assistant"):
        st.subheader(f"✨ {t('result_header')}")
        st.success(f"**{trans_top}** ({results[0][1]*100:.1f}%)")
        
        # --- AUTOMATED TTS OUTPUT ---
        diag_speech = translate_dynamic(f"Based on our conversation, I've identified signs correlated with {trans_top}. Here's your personalized care guide.", st.session_state.lang)
        audio_data = speak_text(diag_speech, st.session_state.lang)
        if audio_data:
            st.audio(audio_data, format="audio/mp3", autoplay=True)
        
        # --- 3D AVATAR (Synchronized Pulse) ---
        target = top_d.lower()
        h_part = "Head" if any(x in target for x in ['cold', 'flu', 'migraine', 'hypertension', 'eye', 'glaucoma']) else "Abdomen"
        st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
        render_3d_avatar(h_part, is_speaking=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write(t('guidance_msg'))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📚 {t('jargon_btn')}"):
                st.info(f"**Simple Terms:** {t_jargon(top_d)}")
            
            # --- PROFESSIONAL PDF DOWNLOAD ---
            pdf_bytes = generate_pdf(st.session_state.get("bio_data", {}), 
                                     st.session_state.selected_symptoms, 
                                     results, 
                                     st.session_state.lang,
                                     shap_summary,
                                     h_part)
            st.download_button(label=t('download_report'), 
                               data=pdf_bytes, 
                               file_name=f"BIOPREDICT_REPORT_{st.session_state.profile_id}.pdf",
                               mime="application/pdf")
        
        with col2:
            st.write(f"**{t('next_steps_header')}:**")
            st.write(t('checklist_item_1'))
            if "Head" in top_d or "Migraine" in top_d: 
                st.write(t('checklist_item_2_head'))
            st.write(t('checklist_item_3'))

        if st.button(t('new_consult_btn')):
            log_health_check(st.session_state.profile_id, st.session_state.selected_symptoms, trans_top)
            st.session_state.step = 0
            st.session_state.selected_symptoms = set()
            st.session_state.chat_log = []
            st.session_state.probing_count = 0
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("AI Health Companion | Built for Global Accessibility | 30+ Languages | Senior Health-Tech UX")
