

import streamlit as st
import pandas as pd
import sqlite3
import requests
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from deep_translator import GoogleTranslator


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PrajaMitra AI",
    page_icon="🏛️",
    layout="wide"
)


# ==========================================
# 2. DATABASE CONNECTION
# ==========================================

def get_connection():
    return sqlite3.connect("prajamitra.db")

def translate_text_batch(texts, language):

    if language == "English":
        return texts

    target_languages = {
        "English": "en",
        "తెలుగు": "te",
        "हिंदी": "hi"
    }

    target_code = target_languages.get(language, "en")

    translated_texts = []

    for text in texts:

        try:
            response = requests.get(
                "https://api.mymemory.translated.net/get",
                params={
                    "q": text,
                    "langpair": f"en|{target_code}"
                },
                timeout=15
            )

            data = response.json()

            if data.get("responseStatus") != 200:
                raise Exception(data.get("responseDetails"))

            translated_texts.append(
                data["responseData"]["translatedText"]
            )

        except Exception as e:
            st.error(f"Translation Error: {e}")
            raise e

    return translated_texts
    target_languages = {
        "English": "en",
        "తెలుగు": "te",
        "हिंदी": "hi"
    }

    target_code = target_languages.get(language, "en")

    try:
        translator = GoogleTranslator(
            source="auto",
            target=target_code
        )

        translated = translator.translate_batch(texts)
        st.write("Translation response:", translated)

        return translated

    except Exception as e:
        st.warning("Translation unavailable. Original text will be saved.")
        return texts


def create_table():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            ID TEXT PRIMARY KEY,
            Name TEXT,
            Mobile TEXT,
            Category TEXT,
            Location TEXT,
            Description TEXT,
            Date TEXT,
            Status TEXT
        )
    """)

    # DATABASE MIGRATION
    # Existing data will be preserved.

    columns = [
        "Original_Name TEXT",
        "Original_Location TEXT",
        "Original_Description TEXT",
        "Citizen_Category TEXT"
    ]

    existing_columns = [
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(complaints)"
        ).fetchall()
    ]

    for column in columns:

        column_name = column.split()[0]

        if column_name not in existing_columns:

            conn.execute(
                f"ALTER TABLE complaints ADD COLUMN {column}"
            )

    conn.commit()
    conn.close()


create_table()


# ==========================================
# 3. AI TRAINING DATA
# ==========================================

training_complaints = [

    # WATER
    "There is a water shortage in our area",
    "Water pipeline is leaking",
    "No drinking water supply",
    "Water is not coming properly",
    "Dirty water is supplied",
    "Water tank is not working",

    # ELECTRICITY
    "Street lights are not working",
    "Electricity is not available",
    "Power cuts happening frequently",
    "Electricity pole is damaged",
    "Transformer is not working",
    "Frequent power failures",

    # ROAD
    "Roads are damaged",
    "There are potholes on the road",
    "Road construction is incomplete",
    "Road is full of holes",
    "Street road is broken",
    "Road repair is needed",

    # SANITATION
    "Garbage is not collected",
    "Drainage water is overflowing",
    "Street is very dirty",
    "Public toilets are not clean",
    "Garbage is dumped on roads",
    "Drainage is blocked",

    # HEALTHCARE
    "Government hospital has no medicines",
    "Hospital staff are not responding",
    "Hospital has no doctors",
    "Medical facilities are not available",
    "Government hospital is not clean",

    # PUBLIC SAFETY
    "There are no street lights near the road",
    "Stray dogs are attacking people",
    "Open manhole is dangerous",
    "Broken electric wires are on the road",
    "Unsafe public area",

    # OTHER
    "I have a problem with government services",
    "I want to report a public issue",
    "Other civic problem"
]


training_categories = [

    "Water", "Water", "Water",
    "Water", "Water", "Water",

    "Electricity", "Electricity", "Electricity",
    "Electricity", "Electricity", "Electricity",

    "Road", "Road", "Road",
    "Road", "Road", "Road",

    "Sanitation", "Sanitation", "Sanitation",
    "Sanitation", "Sanitation", "Sanitation",

    "Healthcare", "Healthcare", "Healthcare",
    "Healthcare", "Healthcare",

    "Public Safety", "Public Safety", "Public Safety",
    "Public Safety", "Public Safety",

    "Other", "Other", "Other"
]


# ==========================================
# 4. TRAIN AI MODEL
# ==========================================

def train_complaint_model():

    model_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    X = model_vectorizer.fit_transform(
        training_complaints
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        training_categories
    )

    return model_vectorizer, model


vectorizer, ai_model = train_complaint_model()


# ==========================================
# 5. AI CATEGORY PREDICTION
# ==========================================

def predict_category(description):

    text = vectorizer.transform([description])

    prediction = ai_model.predict(text)[0]

    return prediction


# ==========================================
# 6. LANGUAGE SUPPORT
# ==========================================

LANGUAGE_CODES = {
    "English": "en",
    "తెలుగు": "te",
    "हिंदी": "hi"
}


def translate_text(text, language):
    if not text or not str(text).strip():
        return text

    target_languages = {
        "English": "en",
        "Telugu": "te",
        "తెలుగు": "te",
        "Hindi": "hi",
        "हिंदी": "hi"
    }

    target_code = target_languages.get(language)

    if not target_code or target_code == "en":
        return text

    try:
        translated = GoogleTranslator(
            source="auto",
            target=target_code
        ).translate(str(text))

        return translated

    except Exception as e:
     st.error(f"Error: {e}")
    st.stop()
# CATEGORY LABELS

CATEGORY_LABELS = {

    "English": {
        "Water": "Water Supply",
        "Electricity": "Electricity",
        "Road": "Roads",
        "Sanitation": "Sanitation",
        "Healthcare": "Healthcare",
        "Public Safety": "Public Safety",
        "Other": "Other"
    },

    "తెలుగు": {
        "Water": "నీటి సరఫరా",
        "Electricity": "విద్యుత్",
        "Road": "రోడ్లు",
        "Sanitation": "పారిశుధ్యం",
        "Healthcare": "ఆరోగ్య సేవలు",
        "Public Safety": "ప్రజా భద్రత",
        "Other": "ఇతర"
    },

    "हिंदी": {
        "Water": "जल आपूर्ति",
        "Electricity": "बिजली",
        "Road": "सड़कें",
        "Sanitation": "स्वच्छता",
        "Healthcare": "स्वास्थ्य सेवाएं",
        "Public Safety": "सार्वजनिक सुरक्षा",
        "Other": "अन्य"
    }
}


# ==========================================
# 7. LANGUAGE TEXT DICTIONARY
# ==========================================

LANGUAGE_TEXT = {

    "English": {

        "home": "Home",
        "register_page": "Register Complaint",
        "track_page": "Track Complaint",
        "dashboard_page": "Dashboard",

        "welcome": "Welcome to PrajaMitra AI 👋",
        "home_desc": "A multilingual AI-powered citizen grievance assistance platform.",

        "register": "Register a Public Grievance",
        "name": "Citizen Name",
        "mobile": "Mobile Number",
        "mobile_hint": "Enter 10-digit mobile number",
        "category": "Complaint Category",
        "location": "Location / Area",
        "date": "Complaint Date",
        "description": "Describe your complaint",
        "submit": "Submit Complaint",

        "fill": "Please fill all fields.",
        "invalid": "Please enter a valid 10-digit mobile number.",
        "duplicate": "This complaint is already registered!",
        "success": "Complaint submitted successfully!",
        "detected": "AI Detected Category:",
        "complaint_id": "Your Complaint ID:",

        "track": "Track Your Complaint",
        "enter_id": "Enter your Complaint Reference ID",
        "check": "Check Status",
        "found": "Complaint Found!",
        "not_found": "Complaint ID not found.",
        "citizen_details": "Citizen Details",
        "complaint_details": "Complaint Details",
        "current_status": "Current Status",
        "mobile_display": "Mobile Number",

        "dashboard": "Complaint Dashboard",
        "total": "Total Complaints",
        "records": "Complaint Records",
        "trends": "Daily Complaint Trends",
        "category_chart": "Complaints by Category",
        "delete": "Delete Complaint",
        "select_id": "Select Complaint ID to Delete",
        "delete_button": "Delete Selected Complaint",
        "deleted": "Complaint deleted successfully!",
        "empty": "No complaints registered yet.",
        "no_dates": "No valid date records available.",
        "mission": "Our mission: Make citizen grievance reporting simple, accessible and transparent.",
        "platform": "Platform",
        "languages": "Languages",
        "register_info": "Submit your public grievance easily.",
        "language_info": "Choose your preferred language.",
        "track_info": "Check your complaint reference ID.",
        "dashboard_info": "View complaint statistics.",
        "platform_value": "AI Assisted",
        "status_submitted": "Submitted",
        "footer": "PrajaMitra AI | Developed by Team CivicNexus | 2026"
    },


    "తెలుగు": {

        "home": "హోమ్",
        "register_page": "ఫిర్యాదు నమోదు",
        "track_page": "ఫిర్యాదు స్థితి",
        "dashboard_page": "డాష్‌బోర్డ్",

        "welcome": "PrajaMitra AI కి స్వాగతం 👋",
        "home_desc": "AI ఆధారిత బహుభాషా పౌరుల ఫిర్యాదుల సహాయ వేదిక.",

        "register": "ప్రజా ఫిర్యాదును నమోదు చేయండి",
        "name": "పౌరుని పేరు",
        "mobile": "మొబైల్ నంబర్",
        "mobile_hint": "10 అంకెల మొబైల్ నంబర్ నమోదు చేయండి",
        "category": "ఫిర్యాదు వర్గం",
        "location": "ప్రాంతం / చిరునామా",
        "date": "ఫిర్యాదు తేదీ",
        "description": "మీ సమస్యను వివరించండి",
        "submit": "ఫిర్యాదును సమర్పించండి",

        "fill": "దయచేసి అన్ని వివరాలను నమోదు చేయండి.",
        "invalid": "సరైన 10 అంకెల మొబైల్ నంబర్ నమోదు చేయండి.",
        "duplicate": "ఈ ఫిర్యాదు ఇప్పటికే నమోదు చేయబడింది!",
        "success": "మీ ఫిర్యాదు విజయవంతంగా నమోదు చేయబడింది!",
        "detected": "AI గుర్తించిన వర్గం:",
        "complaint_id": "మీ ఫిర్యాదు ID:",

        "track": "మీ ఫిర్యాదు స్థితిని తెలుసుకోండి",
        "enter_id": "మీ ఫిర్యాదు ID నమోదు చేయండి",
        "check": "స్థితిని తనిఖీ చేయండి",
        "found": "ఫిర్యాదు కనుగొనబడింది!",
        "not_found": "ఫిర్యాదు ID కనుగొనబడలేదు.",
        "citizen_details": "పౌరుని వివరాలు",
        "complaint_details": "ఫిర్యాదు వివరాలు",
        "current_status": "ప్రస్తుత స్థితి",
        "mobile_display": "మొబైల్ నంబర్",

        "dashboard": "ఫిర్యాదుల డాష్‌బోర్డ్",
        "total": "మొత్తం ఫిర్యాదులు",
        "records": "ఫిర్యాదుల వివరాలు",
        "trends": "రోజువారీ ఫిర్యాదులు",
        "category_chart": "వర్గాల వారీగా ఫిర్యాదులు",
        "delete": "ఫిర్యాదును తొలగించండి",
        "select_id": "తొలగించాల్సిన ఫిర్యాదు ID ఎంచుకోండి",
        "delete_button": "ఎంచుకున్న ఫిర్యాదును తొలగించండి",
        "deleted": "ఫిర్యాదు విజయవంతంగా తొలగించబడింది!",
        "empty": "ఇంకా ఫిర్యాదులు నమోదు కాలేదు.",
        "no_dates": "సరైన తేదీ వివరాలు లేవు.",
        "mission": "మా లక్ష్యం: పౌరుల ఫిర్యాదుల నమోదును సులభంగా మరియు పారదర్శకంగా చేయడం.",
        "platform": "వేదిక",
        "languages": "భాషలు",
        "register_info": "మీ ప్రజా సమస్యను సులభంగా నమోదు చేయండి.",
        "language_info": "మీకు నచ్చిన భాషను ఎంచుకోండి.",
        "track_info": "మీ ఫిర్యాదు ID ద్వారా స్థితిని తెలుసుకోండి.",
        "dashboard_info": "ఫిర్యాదుల గణాంకాలను చూడండి.",
        "platform_value": "AI ఆధారితం",
        "status_submitted": "నమోదు చేయబడింది",
        "footer": "PrajaMitra AI | CivicNexus బృందం | 2026"
    },


    "हिंदी": {

        "home": "होम",
        "register_page": "शिकायत दर्ज करें",
        "track_page": "शिकायत ट्रैक करें",
        "dashboard_page": "डैशबोर्ड",

        "welcome": "PrajaMitra AI में आपका स्वागत है 👋",
        "home_desc": "AI आधारित बहुभाषी नागरिक शिकायत सहायता मंच।",

        "register": "जन शिकायत दर्ज करें",
        "name": "नागरिक का नाम",
        "mobile": "मोबाइल नंबर",
        "mobile_hint": "10 अंकों का मोबाइल नंबर दर्ज करें",
        "category": "शिकायत की श्रेणी",
        "location": "स्थान / क्षेत्र",
        "date": "शिकायत की तारीख",
        "description": "अपनी शिकायत का विवरण दें",
        "submit": "शिकायत दर्ज करें",

        "fill": "कृपया सभी जानकारी भरें।",
        "invalid": "सही 10 अंकों का मोबाइल नंबर दर्ज करें।",
        "duplicate": "यह शिकायत पहले से दर्ज है!",
        "success": "आपकी शिकायत सफलतापूर्वक दर्ज की गई!",
        "detected": "AI द्वारा पहचानी गई श्रेणी:",
        "complaint_id": "आपकी शिकायत ID:",

        "track": "अपनी शिकायत की स्थिति जानें",
        "enter_id": "अपनी शिकायत ID दर्ज करें",
        "check": "स्थिति जांचें",
        "found": "शिकायत मिल गई!",
        "not_found": "शिकायत ID नहीं मिली।",
        "citizen_details": "नागरिक विवरण",
        "complaint_details": "शिकायत विवरण",
        "current_status": "वर्तमान स्थिति",
        "mobile_display": "मोबाइल नंबर",

        "dashboard": "शिकायत डैशबोर्ड",
        "total": "कुल शिकायतें",
        "records": "शिकायत रिकॉर्ड",
        "trends": "दैनिक शिकायत रुझान",
        "category_chart": "श्रेणी के अनुसार शिकायतें",
        "delete": "शिकायत हटाएं",
        "select_id": "हटाने के लिए शिकायत ID चुनें",
        "delete_button": "चयनित शिकायत हटाएं",
        "deleted": "शिकायत सफलतापूर्वक हटा दी गई!",
        "empty": "अभी तक कोई शिकायत दर्ज नहीं हुई।",
        "no_dates": "कोई मान्य तारीख रिकॉर्ड उपलब्ध नहीं है।",
        "mission": "हमारा उद्देश्य: नागरिक शिकायतों को आसान और पारदर्शी बनाना।",
        "platform": "प्लेटफॉर्म",
        "languages": "भाषाएं",
        "register_info": "अपनी सार्वजनिक शिकायत आसानी से दर्ज करें।",
        "language_info": "अपनी पसंदीदा भाषा चुनें।",
        "track_info": "अपनी शिकायत ID से स्थिति जांचें।",
        "dashboard_info": "शिकायतों के आंकड़े देखें।",
        "platform_value": "AI आधारित",
        "status_submitted": "दर्ज किया गया",
        "footer": "PrajaMitra AI | CivicNexus टीम | 2026"
    }
}


# ==========================================
# 8. SIDEBAR AND LANGUAGE SELECTION
# ==========================================

st.sidebar.title("🏛️ PrajaMitra AI")
st.sidebar.caption("CivicNexus | Human Impact")

language = st.sidebar.selectbox(
    "Select Language / భాష ఎంచుకోండి",
    ["English", "తెలుగు", "हिंदी"]
)

T = LANGUAGE_TEXT[language]
C = CATEGORY_LABELS[language]


page_options = [
    T["home"],
    T["register_page"],
    T["track_page"],
    T["dashboard_page"]
]

selected_page = st.sidebar.radio(
    "Navigation / నావిగేషన్",
    page_options
)

# Convert displayed page to internal page name

page_mapping = {
    T["home"]: "Home",
    T["register_page"]: "Register Complaint",
    T["track_page"]: "Track Complaint",
    T["dashboard_page"]: "Dashboard"
}

page = page_mapping[selected_page]


# ==========================================
# 9. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #173b68;
}

.subtitle {
    color: #52657c;
    font-size: 18px;
}

.stButton button {
    background-color: #174b85;
    color: white;
    border-radius: 8px;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 10. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🏛️ PrajaMitra AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your Voice. Our Responsibility.</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 11. HOME PAGE
# ==========================================

if page == "Home":

    st.subheader(T["welcome"])

    st.write(T["home_desc"])

    conn = get_connection()

    total_complaints = conn.execute(
        "SELECT COUNT(*) FROM complaints"
    ).fetchone()[0]

    conn.close()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        T["total"],
        total_complaints
    )

    c2.metric(
        T["platform"],
        T["platform_value"]
    )

    c3.metric(
        T["languages"],
        "3"
    )

    st.info(T["mission"])

    st.subheader("What can you do?")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📝 " + T["register_page"])

        st.write(T["register_info"])

        st.markdown("### 🌐 Multilingual Support")

        st.write(T["language_info"])

    with col2:

        st.markdown("### 🔎 " + T["track_page"])

        st.write(T["track_info"])

        st.markdown("### 📊 " + T["dashboard_page"])

        st.write(T["dashboard_info"])


# ==========================================
# 12. REGISTER COMPLAINT
# ==========================================

elif page == "Register Complaint":

    st.subheader("📝 " + T["register"])

    # Category display to database category mapping
    category_options = list(C.values())

    category_reverse = {
        value: key
        for key, value in C.items()
    }

    name = st.text_input(T["name"])

    mobile = st.text_input(
        T["mobile"],
        max_chars=10,
        placeholder=T["mobile_hint"]
    )

    selected_category = st.selectbox(
        T["category"],
        category_options
    )

    location = st.text_input(T["location"])

    complaint_date = st.date_input(
        T["date"],
        value=datetime.now().date()
    )

    description = st.text_area(
        T["description"],
        height=150
    )

    preview_clicked = st.button("Translate Preview")

    submitted = st.button(T["submit"])

location = st.text_input(T["location"])

complaint_date = st.date_input(
    T["date"],
    value=datetime.now().date()
)

description = st.text_area(
    T["description"],
    height=150
)

preview_clicked = st.button("Translate Preview")

submitted = st.button(T["submit"]) 
if preview_clicked:

    if not name.strip() or not location.strip() or not description.strip():
        st.warning("Please enter Name, Location and Description.")

    else:
        try:
            preview_values = translate_text_batch(
                [
                    name.strip(),
                    location.strip(),
                    description.strip()
                ],
                language
            )

            st.subheader("Translated Preview")

            st.write("Name:", preview_values[0])
            st.write("Location:", preview_values[1])
            st.write("Description:", preview_values[2])

        except Exception as e:
            st.error(f"Preview Translation Error: {e}")      
            if submitted:

        # STEP 1: VALIDATION
             if (
            not name.strip()
            or not mobile.strip()
            or not location.strip()
            or not description.strip()
        ):
              st.error(T["fill"])

             elif not mobile.isdigit() or len(mobile) != 10:
              st.error(T["invalid"])

        else:
            try:

                # STEP 2: TRANSLATION
                translated_values = translate_text_batch(
                    [
                        name.strip(),
                        location.strip(),
                        description.strip()
                    ],
                    language
                )

                translated_name = translated_values[0]
                translated_location = translated_values[1]
                translated_description = translated_values[2]

                english_description = description.strip()

                # STEP 3: AI CATEGORY PREDICTION
                predicted_category = predict_category(
                    english_description
                )

                citizen_category = category_reverse[
                    selected_category
                ]

                # STEP 4: GENERATE UNIQUE COMPLAINT ID
                complaint_id = (
                    "PM" +
                    datetime.now().strftime("%Y%m%d%H%M%S%f")
                )

                # STEP 5: COMPLAINT DATA
                complaint = {
                    "ID": complaint_id,
                    "Name": translated_name,
                    "Mobile": mobile,
                    "Category": predicted_category,
                    "Location": translated_location,
                    "Description": translated_description,
                    "Date": (
                        complaint_date.strftime("%d-%m-%Y")
                        + " "
                        + datetime.now().strftime("%H:%M")
                    ),
                    "Status": "Submitted",
                    "Original_Name": name.strip(),
                    "Original_Location": location.strip(),
                    "Original_Description": description.strip(),
                    "Citizen_Category": citizen_category
                }

                # STEP 6: SAVE TO DATABASE
                conn = get_connection()

                try:
                    conn.execute(
                        """
                        INSERT INTO complaints
                        (
                            ID,
                            Name,
                            Mobile,
                            Category,
                            Location,
                            Description,
                            Date,
                            Status,
                            Original_Name,
                            Original_Location,
                            Original_Description,
                            Citizen_Category
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            complaint["ID"],
                            complaint["Name"],
                            complaint["Mobile"],
                            complaint["Category"],
                            complaint["Location"],
                            complaint["Description"],
                            complaint["Date"],
                            complaint["Status"],
                            complaint["Original_Name"],
                            complaint["Original_Location"],
                            complaint["Original_Description"],
                            complaint["Citizen_Category"]
                        )
                    )

                    conn.commit()

                    st.success(T["success"])

                    st.write(
                        T["detected"],
                        C[predicted_category]
                    )

                    st.write(T["complaint_id"])
                    st.code(complaint_id)

                finally:
                    conn.close()

            except Exception as e:
                st.error(f"Error: {e}")


# ==========================================
# 13. TRACK COMPLAINT
# ========================================== 
elif page == "Track Complaint":

    st.subheader("🔎 " + T["track"])

    complaint_id = st.text_input(
        T["enter_id"]
    )

    if st.button(T["check"]):

        if not complaint_id.strip():

            st.warning(T["enter_id"])

        else:

            conn = get_connection()

            row = conn.execute(
                """
                SELECT
                    ID,
                    Name,
                    Mobile,
                    Category,
                    Location,
                    Description,
                    Date,
                    Status
                FROM complaints
                WHERE ID = ?
                """,

                (complaint_id.strip(),)

            ).fetchone()

            conn.close()


            if row:

                keys = [
                    "ID",
                    "Name",
                    "Mobile",
                    "Category",
                    "Location",
                    "Description",
                    "Date",
                    "Status"
                ]

                complaint = dict(
                    zip(keys, row)
                )

                st.success(T["found"])

                st.subheader(T["citizen_details"])

                st.write(
                    T["name"] + ":",
                    complaint["Name"]
                )

                mobile = complaint["Mobile"]

                masked_mobile = (
                    "******" + mobile[-4:]
                    if len(mobile) == 10
                    else mobile
                )

                st.write(
                    T["mobile_display"] + ":",
                    masked_mobile
                )

                st.subheader(T["complaint_details"])

                st.write(
                    T["category"] + ":",
                    C.get(
                        complaint["Category"],
                        complaint["Category"]
                    )
                )

                st.write(
                    T["location"] + ":",
                    complaint["Location"]
                )

                st.write(
                    T["description"] + ":",
                    complaint["Description"]
                )

                st.write(
                    T["date"] + ":",
                    complaint["Date"]
                )

                status = complaint["Status"]

                if status == "Submitted":

                    status = T["status_submitted"]

                st.info(
                    T["current_status"] + ": " + status
                )

            else:

                st.error(T["not_found"])


# ==========================================
# 14. DASHBOARD
# ==========================================

elif page == "Dashboard":

    st.subheader("📊 " + T["dashboard"])

    # GET COMPLAINTS FROM DATABASE

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM complaints",
        conn
    )

    conn.close()


    st.metric(
        T["total"],
        len(df)
    )


    if not df.empty:

        # COMPLAINT RECORDS

        st.subheader("📋 " + T["records"])

        display_df = df[
            [
                "ID",
                "Name",
                "Mobile",
                "Category",
                "Location",
                "Description",
                "Date",
                "Status"
            ]
        ].copy()

        display_df["Category"] = (
            display_df["Category"].map(
                lambda x: C.get(x, x)
            )
        )

        display_df["Status"] = (
            display_df["Status"].map(
                lambda x: (
                    T["status_submitted"]
                    if x == "Submitted"
                    else x
                )
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )


        # DAILY COMPLAINT TRENDS

        st.subheader("📈 " + T["trends"])

        df["ParsedDate"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce"
        )

        valid_df = df.dropna(
            subset=["ParsedDate"]
        ).copy()


        if not valid_df.empty:

            daily_counts = (
                valid_df.groupby(
                    valid_df["ParsedDate"].dt.date
                )
                .size()
                .sort_index()
            )

            daily_counts.index = [
                date.strftime("%d-%m-%Y")
                for date in daily_counts.index
            ]

            st.line_chart(daily_counts)

        else:

            st.warning(T["no_dates"])


        # CATEGORY WISE BAR CHART

        st.subheader("📊 " + T["category_chart"])

        category_counts = (
            df["Category"]
            .map(lambda x: C.get(x, x))
            .value_counts()
        )

        st.bar_chart(category_counts)


        # DELETE COMPLAINT

        st.subheader("🗑️ " + T["delete"])

        complaint_ids = df["ID"].tolist()

        selected_id = st.selectbox(
            T["select_id"],
            complaint_ids
        )

        if st.button(T["delete_button"]):

            conn = get_connection()

            conn.execute(
                "DELETE FROM complaints WHERE ID = ?",
                (selected_id,)
            )

            conn.commit()
            conn.close()

            st.success(T["deleted"])

            st.rerun()


    else:

        st.info(T["empty"])


# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.caption(T["footer"])
   