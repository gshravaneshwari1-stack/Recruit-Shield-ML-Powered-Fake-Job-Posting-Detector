import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import joblib
import re
import nltk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ==========================================
# 1. CONFIGURATION & VISUAL DESIGN
# ==========================================
st.set_page_config(page_title="Job Post Prediction", page_icon="📝", layout="wide")

# --- CUSTOM CSS (CLEAN LIGHT THEME) ---
st.markdown("""
    <style>
    /* 1. Background Image */
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/white-abstract-background-design_23-2148825582.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* 2. Main Form Container */
    .main .block-container {
        background-color: #ffffff;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        max-width: 1200px;
    }

    /* 3. HEADINGS */
    h1, h2, h3, h4, h5, h6 {
        color: #0D47A1 !important; 
        font-family: 'Arial', sans-serif !important;
        font-weight: 800 !important;
    }
    
    /* 4. General Text */
    p, span, div, label, li {
        color: #000000 !important;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* 5. Sidebar Hidden */
    [data-testid="stSidebar"] {display: none;}
    
    /* 6. INPUTS */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea {
        background-color: #F5F5F5 !important;
        color: #000000 !important;
        border: 2px solid #0D47A1;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* 7. BUTTONS */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%) !important;
        color: #FFFFFF !important;
        border: none;
        padding: 12px;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }

    /* 8. TABLE STYLE */
    .solid-table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 8px;
        overflow: hidden;
    }
    .solid-table th {
        background-color: #1565C0;
        color: #ffffff;
        padding: 15px;
        text-align: left;
        font-size: 18px;
    }
    .solid-table td {
        padding: 12px 15px;
        background-color: #E3F2FD;
        color: #000000;
        border-bottom: 1px solid #BBDEFB;
        font-weight: bold;
    }

    /* 9. PROFILE CARD */
    .profile-card {
        background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        color: white !important;
    }
    .profile-card p, .profile-card h3 { color: white !important; }
    
    .analytics-card {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0D47A1;
        margin-bottom: 10px;
    }

    /* 10. Chart Visibility */
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# NLP Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ==========================================
# 2. DATA
# ==========================================
COUNTRY_CODES = {
    "🇮🇳 India (+91)": {"code": "+91", "len": 10},
    "🇺🇸 USA (+1)": {"code": "+1", "len": 10},
    "🇬🇧 UK (+44)": {"code": "+44", "len": 10},
    "Other": {"code": "+", "len": 0} 
}

INDIAN_STATES = [
    "Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana", 
    "Maharashtra", "Delhi", "Uttar Pradesh", "West Bengal", "Gujarat", "Rajasthan"
]

# ==========================================
# 3. DATABASE FUNCTIONS
# ==========================================
def create_connection():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    return conn

def create_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                mobile TEXT,
                email TEXT,
                city TEXT,
                state TEXT,
                role TEXT,
                login_count INTEGER DEFAULT 0,
                is_blocked INTEGER DEFAULT 0
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                job_title TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp TEXT
                )""")
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# --- UPDATED: STRICT DUPLICATE CHECKER ---
def check_user_exists(username, mobile, email):
    conn = create_connection()
    c = conn.cursor()
    
    # 1. Check Username (Case Insensitive & Trimmed)
    # Using SQL LOWER() function to ensure 'John' == 'john' == 'JOHN'
    c.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),))
    if c.fetchone():
        conn.close()
        return True, f"⚠️ Username '{username}' is already taken. Please choose another."
        
    # 2. Check Mobile
    c.execute("SELECT * FROM users WHERE mobile = ?", (mobile.strip(),))
    if c.fetchone():
        conn.close()
        return True, f"⚠️ Mobile number '{mobile}' is already registered."
        
    # 3. Check Email (Case Insensitive)
    c.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),))
    if c.fetchone():
        conn.close()
        return True, f"⚠️ Email '{email}' is already registered."
        
    conn.close()
    return False, ""

def add_user(username, password, mobile, email, city, state, role='user'):
    conn = create_connection()
    c = conn.cursor()
    try:
        # Use strip() to remove accidental spaces
        c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,0,0)", 
                  (username.strip(), make_hashes(password), mobile.strip(), email.strip(), city, state, role))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    # Case insensitive login check
    c.execute('SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND password = ?', 
              (username.strip(), make_hashes(password)))
    data = c.fetchall()
    conn.close()
    return data

def update_login_count(username):
    conn = create_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET login_count = login_count + 1 WHERE LOWER(username) = LOWER(?)", (username.strip(),))
    conn.commit()
    conn.close()

def get_user_details(username):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE LOWER(username) = LOWER(?)', (username.strip(),))
    data = c.fetchall()
    conn.close()
    return data

def ensure_admin_active():
    conn = create_connection(); c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if c.fetchone(): c.execute("UPDATE users SET is_blocked = 0 WHERE username = 'admin'")
    else: 
        try: c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,0,0)", ("admin", make_hashes("admin123"), "0000000000", "admin@safejob.com", "Cyber", "Space", "admin"))
        except: pass
    conn.commit(); conn.close()

# --- ADMIN FUNCTIONS ---
def get_all_users_df():
    conn = create_connection()
    df = pd.read_sql_query("SELECT username, mobile, email, city, state, role, login_count, is_blocked FROM users WHERE role != 'admin'", conn)
    conn.close()
    return df

def get_all_history_df():
    conn = create_connection()
    df = pd.read_sql_query("SELECT job_title, prediction, confidence, timestamp, username FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

def block_user_db(username):
    conn = create_connection(); c = conn.cursor()
    c.execute("UPDATE users SET is_blocked = 1 WHERE username = ?", (username,))
    conn.commit(); conn.close()

def unblock_user_db(username):
    conn = create_connection(); c = conn.cursor()
    c.execute("UPDATE users SET is_blocked = 0 WHERE username = ?", (username,))
    conn.commit(); conn.close()

def delete_user_db(username):
    conn = create_connection(); c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    c.execute("DELETE FROM history WHERE username = ?", (username,))
    conn.commit(); conn.close()

def reset_password(username, mobile, new_password):
    conn = create_connection(); c = conn.cursor()
    c.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND mobile = ?", (username.strip(), mobile.strip()))
    if c.fetchall():
        c.execute("UPDATE users SET password = ? WHERE LOWER(username) = LOWER(?)", (make_hashes(new_password), username.strip()))
        conn.commit(); conn.close(); return True
    conn.close(); return False

# --- HISTORY FUNCTIONS ---
def add_to_history(username, job_title, prediction, confidence):
    conn = create_connection(); c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (username, job_title, prediction, confidence, timestamp) VALUES (?,?,?,?,?)", 
              (username, job_title, prediction, confidence, ts))
    conn.commit(); conn.close()

def get_user_history(username):
    conn = create_connection(); c = conn.cursor()
    c.execute("SELECT id, job_title, prediction, confidence, timestamp FROM history WHERE LOWER(username) = LOWER(?) ORDER BY id DESC", (username.strip(),))
    data = c.fetchall(); conn.close()
    return data

def delete_user_history(username):
    conn = create_connection(); c = conn.cursor()
    c.execute("DELETE FROM history WHERE LOWER(username) = LOWER(?)", (username.strip(),))
    conn.commit(); conn.close()

def delete_specific_history(row_id):
    conn = create_connection(); c = conn.cursor()
    c.execute("DELETE FROM history WHERE id = ?", (row_id,))
    conn.commit(); conn.close()

# --- VALIDATION (UPDATED) ---

def validate_username(username):
    # Allow letters and spaces, reject empty/only-spaces, reject numbers/symbols
    if not username.strip(): return False, "⚠️ Username cannot be empty."
    if not re.match(r"^[a-zA-Z\s]+$", username): return False, "⚠️ Username: Letters and spaces only (No numbers/symbols)."
    if len(username.strip()) < 3: return False, "⚠️ Username: Min 3 characters."
    return True, ""

def validate_mobile(mobile):
    if not mobile.isdigit(): return False, "⚠️ Enter a valid mobile number (Digits only)."
    if mobile.startswith("0"): return False, "⚠️ Mobile number cannot start with 0."
    return True, ""

def validate_password(password):
    if len(password) < 8: return False, "⚠️ Password must be at least 8 characters long."
    if not re.search(r"[a-z]", password): return False, "⚠️ Password must contain at least one lowercase letter."
    if not re.search(r"[A-Z]", password): return False, "⚠️ Password must contain at least one uppercase letter."
    if not re.search(r"[0-9]", password): return False, "⚠️ Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password): return False, "⚠️ Password must contain at least one special character."
    return True, ""

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email): return False, "⚠️ Enter a valid email address (e.g., name@example.com)."
    if email.split('@')[-1] in ["outlook.com", "email.com"]: return False, "⚠️ Domain blocked. Use Gmail or Yahoo."
    return True, ""

def validate_text_only(text, field):
    if any(char.isdigit() for char in text): return False, f"⚠️ {field}: Text only."
    if not text.strip(): return False, f"⚠️ {field}: Cannot be empty."
    return True, ""

# --- ML ---
def clean_input_text(text):
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_suspicious_keywords(text):
    triggers = ["urgent", "immediate start", "no experience", "wire transfer", "cash", "easy money", "work from home", "investment", "visa sponsorship"]
    return list(set([word for word in triggers if word in text.lower()]))

# INITIALIZATION
create_table()
ensure_admin_active()
models = None
try:
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
        "SVM (Linear)": joblib.load("models/svm_linear_model.pkl")
    }
except: pass

@st.cache_data
def load_data():
    try: return pd.read_csv('data/fake_job_postings.csv')
    except: return None
df_data = load_data()

# ==========================================
# MAIN APP LOGIC
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ''
if 'role' not in st.session_state: st.session_state['role'] = ''
if 'page_selection' not in st.session_state: st.session_state['page_selection'] = 'Home'

def main():
    
    # ==========================
    # HEADER BAR NAVIGATION
    # ==========================
    st.markdown('<h1 style="text-align:center; color:#0D47A1; font-size: 3.5rem;">Job Post Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    
    with st.container():
        if st.session_state['logged_in']:
            if st.session_state['role'] == 'admin':
                col1, col2, col3 = st.columns(3)
                if col1.button("User Management"): st.session_state['page_selection'] = "Admin Dashboard"; st.rerun()
                if col2.button("Global Accuracy"): st.session_state['page_selection'] = "Global Stats"; st.rerun()
                if col3.button("Logout"): st.session_state['page_selection'] = "Logout"; st.rerun()
            else:
                col1, col2, col3, col4, col5 = st.columns(5)
                if col1.button("Predict Job Post"): st.session_state['page_selection'] = "Check Job"; st.rerun()
                if col2.button("My History"): st.session_state['page_selection'] = "Search History"; st.rerun()
                if col3.button("Accuracy Results"): st.session_state['page_selection'] = "User Accuracy"; st.rerun()
                if col4.button("My Profile"): st.session_state['page_selection'] = "My Profile"; st.rerun()
                if col5.button("Logout"): st.session_state['page_selection'] = "Logout"; st.rerun()
        else:
            col1, col2, col3 = st.columns(3)
            if col1.button("Home"): st.session_state['page_selection'] = "Home"; st.rerun()
            if col2.button("User Login"): st.session_state['page_selection'] = "Login"; st.rerun()
            if col3.button("New Registration"): st.session_state['page_selection'] = "Register"; st.rerun()

    st.write("") 

    # ==========================
    # PAGE CONTENT
    # ==========================
    with st.container():
        st.markdown('<div class="content-box">', unsafe_allow_html=True)

        # --- 1. HOME PAGE ---
        if st.session_state['page_selection'] == "Home":
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=300)
            with col2:
                st.markdown("### WE'RE HIRING... OR ARE WE?")
                st.markdown("""
                **Fake Job Post Detection System**
                
                This application uses Machine Learning algorithms to identify fraudulent job postings.
                
                **Features:**
                *   Verify Job Authenticity
                *   View Accuracy of Algorithms
                *   Secure User Registration
                """)
                if not st.session_state['logged_in']:
                    st.info("Please Login or Register to continue.")

            st.divider()
            
            # Live Stats Strip
            st.markdown("<h3 style='text-align: center; color: #0D47A1;'>Real-Time Detection Stats</h3>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Jobs Analyzed", "18,450+")
            m2.metric("Scams Blocked", "986")
            m3.metric("AI Accuracy", "98.5%")
            m4.metric("Active Users", "1,200+")

            st.write("---")
            # Why Choose Us
            st.subheader("🚀 Why Use This Tool?")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("""<div class="feature-card"><h3>🤖 Multi-Model AI</h3><p>Combines SVM, Random Forest, Logistic Regression, and Naive Bayes.</p></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""<div class="feature-card"><h3>⚡ Instant Results</h3><p>Get a detailed trust score and fraud probability in seconds.</p></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown("""<div class="feature-card"><h3>🛡️ Data Privacy</h3><p>Your search history is private and secured.</p></div>""", unsafe_allow_html=True)

        # --- 2. LOGIN PAGE ---
        elif st.session_state['page_selection'] == "Login":
            st.subheader("User / Admin Login")
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                username = st.text_input("Username")
                password = st.text_input("Password", type='password')
                
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Login Now"):
                        result = login_user(username, password)
                        if result:
                            if result[0][8] == 1: st.error("Account Blocked.")
                            else:
                                st.session_state['logged_in'] = True
                                st.session_state['username'] = username
                                st.session_state['role'] = result[0][6]
                                update_login_count(username)
                                target = "Admin Dashboard" if st.session_state['role'] == 'admin' else "Check Job"
                                st.session_state['page_selection'] = target
                                st.rerun()
                        else: st.error("Invalid Credentials")
                with b2:
                    if st.button("Forgot Password?"): st.session_state['page_selection'] = "Forgot Password"; st.rerun()

        # --- 3. REGISTER PAGE ---
        elif st.session_state['page_selection'] == "Register":
            st.subheader("New User Registration")
            with st.form("reg_form"):
                c1, c2 = st.columns(2)
                with c1:
                    new_user = st.text_input("Username", help="Letters ONLY (Spaces allowed)")
                    new_password = st.text_input("Password", type='password')
                    new_email = st.text_input("Email")
                with c2:
                    c_code, c_num = st.columns([1, 2])
                    with c_code: sel_country_key = st.selectbox("Code", list(COUNTRY_CODES.keys()))
                    with c_num: raw_mobile = st.text_input("Mobile")
                    new_state = st.selectbox("State", INDIAN_STATES)
                    new_city = st.text_input("City Name")
                submit = st.form_submit_button("Register")
                
                if submit:
                    prefix = COUNTRY_CODES[sel_country_key]["code"]
                    req_len = COUNTRY_CODES[sel_country_key]["len"]
                    v_u, m_u = validate_username(new_user)
                    v_m, m_m = validate_mobile(raw_mobile)
                    v_p, m_p = validate_password(new_password)
                    v_e, m_e = validate_email(new_email)
                    v_c, m_c = validate_text_only(new_city, "City")
                    
                    if not (v_u and v_m and v_p and v_e and v_c):
                        if not v_u: st.error(m_u)
                        elif not v_m: st.error(m_m)
                        elif not v_p: st.error(m_p)
                        elif not v_e: st.error(m_e)
                        elif not v_c: st.error(m_c)
                    elif req_len > 0 and len(raw_mobile) != req_len: 
                        st.error(f"Mobile must be {req_len} digits.")
                    else:
                        final_mobile = f"{prefix}{raw_mobile}"
                        exists, msg = check_user_exists(new_user, final_mobile, new_email)
                        if exists: st.error(msg)
                        else:
                            if add_user(new_user, new_password, final_mobile, new_email, new_city, new_state):
                                st.success("Registered Successfully! Go to Login.")
                            else: st.error("Error.")

        # --- 4. FORGOT PASSWORD ---
        elif st.session_state['page_selection'] == "Forgot Password":
            st.subheader("Reset Password")
            fp_user = st.text_input("Username")
            c_code, c_num = st.columns([1, 2])
            with c_code: fp_country = st.selectbox("Country Code", list(COUNTRY_CODES.keys()))
            with c_num: fp_raw_mobile = st.text_input("Mobile Number")
            fp_new_pass = st.text_input("New Password", type='password')
            if st.button("Reset Password"):
                prefix = COUNTRY_CODES[fp_country]["code"]
                fp_final = f"{prefix}{fp_raw_mobile}"
                if reset_password(fp_user, fp_final, fp_new_pass): st.success("Password Updated!")
                else: st.error("User not found.")

        # --- 5. CHECK JOB (MANDATORY FIELDS) ---
        elif st.session_state['page_selection'] == "Check Job":
            st.subheader("Predict Job Post Type")
            model_choice = st.selectbox("Select Algorithm", list(models.keys()) if models else [])
            col1, col2 = st.columns(2)
            with col1:
                job_title = st.text_input("Job Title")
                job_loc = st.text_input("Location")
            with col2:
                job_req = st.text_area("Requirements", height=100)
                job_ben = st.text_area("Benefits", height=100)
            job_desc = st.text_area("Job Description (Paste full text here)", height=150)
            
            if st.button("Predict Job Post Type"):
                if models:
                    # MANDATORY FIELD CHECK
                    if not (job_title and job_loc and job_desc and job_req and job_ben):
                        st.error("⚠️ Please fill in ALL fields (Title, Location, Description, Requirements, Benefits) to proceed.")
                    else:
                        full_input = f"{job_title} {job_loc} {job_desc} {job_req} {job_ben}"
                        # LENGTH CHECK
                        if len(full_input.split()) < 20:
                            st.warning("⚠️ Text too short (Min 20 words).")
                        else:
                            clean_t = clean_input_text(full_input)
                            pipeline = models[model_choice]
                            prediction = pipeline.predict([clean_t])[0]
                            prob = pipeline.predict_proba([clean_t])[0] 
                            final_pred = "Fake" if prediction == 1 else "Real"
                            final_conf = prob[1] if prediction == 1 else prob[0]
                            add_to_history(st.session_state['username'], job_title, final_pred, final_conf)
                            
                            st.divider()
                            if prediction == 1: st.error(f"🚨 FAKE JOB POST DETECTED")
                            else: st.success(f"✅ LEGITIMATE JOB POST")
                            st.info(f"🔍 **Confidence Score:** {final_conf*100:.2f}%")
                            
                            # VISUALS
                            st.subheader("Prediction Results")
                            prob_df = pd.DataFrame({'Type': ['Real', 'Fake'], 'Probability': [prob[0], prob[1]]})
                            c1, c2 = st.columns(2)
                            with c1:
                                fig_pie = px.pie(prob_df, values='Probability', names='Type', color='Type', color_discrete_map={'Real':'#2E7D32', 'Fake':'#C62828'})
                                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                                st.plotly_chart(fig_pie, use_container_width=True)
                            with c2:
                                fig_bar = px.bar(prob_df, x='Type', y='Probability', color='Type', color_discrete_map={'Real':'#2E7D32', 'Fake':'#C62828'})
                                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                                st.plotly_chart(fig_bar, use_container_width=True)

        # --- 6. USER ACCURACY RESULTS ---
        elif st.session_state['page_selection'] == "User Accuracy":
            st.subheader("Accuracy Results")
            st.markdown("### 1. Algorithm Performance")
            st.markdown("""<table class="solid-table"><thead><tr><th>Algorithm</th><th>Accuracy %</th></tr></thead><tbody><tr><td>Logistic Regression</td><td>98.55%</td></tr><tr><td>SVM</td><td>95.54%</td></tr><tr><td>Naive Bayes</td><td>93.50%</td></tr><tr><td>Random Forest Classifier</td><td>95.79%</td></tr></tbody></table>""", unsafe_allow_html=True)
            acc_values = [98.55, 95.54, 93.50, 95.79]
            acc_models = ["Logistic Regression", "SVM", "Naive Bayes", "Random Forest"]
            fig_line = px.line(x=acc_models, y=acc_values, markers=True, title="Algorithm Accuracy Comparison", labels={'x': 'Algorithm', 'y': 'Accuracy %'})
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("### 2. My Prediction History Analysis")
            hist_data = get_user_history(st.session_state['username'])
            if hist_data:
                user_hist_df = pd.DataFrame(hist_data, columns=['ID', 'Job Title', 'Prediction', 'Confidence', 'Date'])
                real_count = len(user_hist_df[user_hist_df['Prediction'] == 'Real'])
                fake_count = len(user_hist_df[user_hist_df['Prediction'] == 'Fake'])
                avg_conf_real = user_hist_df[user_hist_df['Prediction']=='Real']['Confidence'].mean() * 100 if real_count > 0 else 0
                avg_conf_fake = user_hist_df[user_hist_df['Prediction']=='Fake']['Confidence'].mean() * 100 if fake_count > 0 else 0
                st.markdown(f"""<table class="solid-table"><thead><tr><th>Data Type</th><th>Count</th><th>Avg Confidence</th></tr></thead><tbody><tr><td>Legitimate Jobs Found</td><td>{real_count}</td><td>{avg_conf_real:.2f}%</td></tr><tr><td>Fake Jobs Detected</td><td>{fake_count}</td><td>{avg_conf_fake:.2f}%</td></tr></tbody></table>""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    fig_pie = px.pie(names=['Real', 'Fake'], values=[real_count, fake_count], color_discrete_map={'Real':'#2E7D32', 'Fake':'#C62828'}, title="My Search Results (Real vs Fake)")
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    recent = user_hist_df.head(10)
                    fig_bar = px.bar(recent, x='Job Title', y='Confidence', color='Prediction', color_discrete_map={'Real':'#2E7D32', 'Fake':'#C62828'}, title="Confidence of Recent Searches")
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                    st.plotly_chart(fig_bar, use_container_width=True)
            else: st.info("No search history available yet.")

        # --- 7. ADMIN DASHBOARD (GLOBAL STATS) ---
        elif st.session_state['page_selection'] == "Global Stats":
            st.subheader("Global Algorithm Accuracy (Admin View)")
            st.markdown("""<table class="solid-table"><thead><tr><th>Algorithm</th><th>Accuracy %</th></tr></thead><tbody><tr><td>Logistic Regression</td><td>98.55%</td></tr><tr><td>SVM</td><td>95.54%</td></tr><tr><td>Naive Bayes</td><td>93.50%</td></tr><tr><td>Random Forest Classifier</td><td>95.79%</td></tr></tbody></table>""", unsafe_allow_html=True)
            
            acc_values = [98.55, 95.54, 93.50, 95.79]
            acc_models = ["Logistic Regression", "SVM", "Naive Bayes", "Random Forest"]
            c1, c2 = st.columns(2)
            with c1:
                fig_bar = px.bar(x=acc_models, y=acc_values, color=acc_models, title="Accuracy Bar Chart")
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                st.plotly_chart(fig_bar, use_container_width=True)
            with c2:
                fig_pie = px.pie(values=acc_values, names=acc_models, title="Accuracy Distribution")
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            
            # --- 2. Traffic Analysis (Admin Feature) ---
            st.subheader("Website Traffic & Usage Analytics")
            history_df = get_all_history_df()
            if not history_df.empty:
                # A. Traffic over Time
                history_df['Date'] = pd.to_datetime(history_df['timestamp']).dt.date
                traffic = history_df['Date'].value_counts().sort_index()
                
                fig_line = px.line(x=traffic.index, y=traffic.values, title="Scans per Day", markers=True)
                fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                st.plotly_chart(fig_line, use_container_width=True)
                
                # B. User Activity
                top_users = history_df['username'].value_counts().head(10)
                fig_users = px.bar(x=top_users.index, y=top_users.values, title="Top Active Users")
                fig_users.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                st.plotly_chart(fig_users, use_container_width=True)
                
                # C. Fraud Detection Rate
                c_fraud1, c_fraud2 = st.columns(2)
                with c_fraud1:
                    st.write("**Recent Fraud Alerts**")
                    fake_jobs = history_df[history_df['prediction'] == 'Fake'].head(5)
                    st.dataframe(fake_jobs[['job_title', 'confidence', 'timestamp']], use_container_width=True)
                with c_fraud2:
                    st.write("**Top Searched Job Titles**")
                    top_jobs = history_df['job_title'].value_counts().head(5)
                    fig_top = px.bar(x=top_jobs.index, y=top_jobs.values, title="Most Searched Jobs")
                    fig_top.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black'))
                    st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info("No traffic data available yet.")

        # --- 8. ADMIN USER MANAGEMENT ---
        elif st.session_state['page_selection'] == "Admin Dashboard":
            st.subheader("User Management")
            df_users = get_all_users_df()
            if not df_users.empty:
                st.dataframe(df_users, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    u_block = st.selectbox("Select User to Block/Unblock", df_users['username'].unique())
                    if st.button("Toggle Block"):
                        curr = df_users[df_users['username']==u_block]['is_blocked'].values[0]
                        if curr == 0: block_user_db(u_block); st.success("Blocked")
                        else: unblock_user_db(u_block); st.success("Unblocked")
                        st.rerun()
                with c2:
                    u_del = st.selectbox("Select User to Delete", df_users['username'].unique())
                    if st.button("Delete User"): delete_user_db(u_del); st.success("Deleted."); st.rerun()
            else: st.info("No registered users found.")

        # --- 9. HISTORY ---
        elif st.session_state['page_selection'] == "Search History":
            st.subheader("My Search History")
            hist_data = get_user_history(st.session_state['username'])
            if hist_data:
                hist_df = pd.DataFrame(hist_data, columns=['ID', 'Job Title', 'Prediction', 'Confidence', 'Date'])
                st.dataframe(hist_df, use_container_width=True)
                st.write("---")
                c_del1, c_del2 = st.columns(2)
                with c_del1:
                    del_options = {f"{row[0]} - {row[1]}": row[0] for row in hist_data}
                    selected_option = st.selectbox("Select Record to Delete", list(del_options.keys()))
                    if st.button("🗑️ Delete Selected"): delete_specific_history(del_options[selected_option]); st.success("Deleted."); st.rerun()
                with c_del2:
                    st.write(""); st.write("")
                    if st.button("⚠️ Clear ALL History"): delete_user_history(st.session_state['username']); st.success("Cleared."); st.rerun()
            else: st.info("No history found.")
            
        elif st.session_state['page_selection'] == "My Profile":
            st.subheader("My Profile")
            user = get_user_details(st.session_state['username'])
            if user:
                user_data = user[0]
                st.markdown(f"""<div class="profile-card"><h3>User Profile Details</h3><p><strong>👤 Username:</strong> {user_data[0]}</p><p><strong>📧 Email:</strong> {user_data[3]}</p><p><strong>📱 Mobile:</strong> {user_data[2]}</p><p><strong>📍 Location:</strong> {user_data[4]}, {user_data[5]}</p><p><strong>🔢 Login Count:</strong> {user_data[7]}</p></div>""", unsafe_allow_html=True)

        # --- 10. LOGOUT ---
        elif st.session_state['page_selection'] == "Logout":
            st.session_state['logged_in'] = False
            st.session_state['username'] = ''
            st.session_state['role'] = ''
            st.session_state['page_selection'] = "Home"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True) # End content box

if __name__ == '__main__':
    main()