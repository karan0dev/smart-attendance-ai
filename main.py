import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
import pickle
import cv2
from datetime import datetime
import face_recognition
from scipy.spatial import distance as dist

# --- INITIALIZATION & SESSION STATE ---
st.set_page_config(page_title="Attendance Desk", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "data/"
ATTENDANCE_DIR = "Attendance/"
CRED_FILE = "data/user_credentials.csv"
ENCODINGS_FILE = "data/encodings.pkl"

# Enforce environment: Ensure Numpy 1.x is running to prevent dlib crash
if os.path.exists(ENCODINGS_FILE):
    import numpy as np 
    if not np.__version__.startswith("1."):
        st.error(f"FATAL SYSTEM ERROR: Incompatible Numpy version detected ({np.__version__}). System requires Numpy 1.26.4. Run 'pip uninstall numpy -y' then 'pip install numpy==1.26.4' in terminal.")
        st.stop()

for path in [DATA_DIR, ATTENDANCE_DIR]:
    if not os.path.exists(path): os.makedirs(path)

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['admin_name'] = "Unknown Admin"
if 'page' not in st.session_state:
    st.session_state['page'] = "Home"
if 'selected_log' not in st.session_state:
    st.session_state['selected_log'] = None

# --- MATHEMATICAL LIVENESS DETECTION ---
def eye_aspect_ratio(eye):
    """Computes the EAR using Euclidean distances between specific eye landmarks."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- CORE BIOMETRIC FUNCTIONS (DEEP LEARNING) ---
def capture_new_student(full_name, roll_no):
    """Extracts a 128-d facial embedding vector instead of raw pixel arrays."""
    video = cv2.VideoCapture(0)
    st.warning("Webcam active. Look at the camera and press 'c' to capture, or 'q' to abort.")
    
    encoded_face = None
    
    while True:
        ret, frame = video.read()
        if not ret: continue
        
        cv2.putText(frame, "Position face clearly. Press 'c' to capture.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Biometric Matrix Enrollment", frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Enforce memory contiguity and 8-bit casting to prevent dlib RuntimeError
            rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype(np.uint8)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) != 1:
                st.toast("Capture Error: Exactly ONE face must be in the frame.")
                continue
                
            # Extract the 128-dimensional embedding
            encoded_face = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            break
            
        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    
    if encoded_face is not None:
        name_entry = f"{full_name} ({roll_no})"
        
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                known_data = pickle.load(f)
        else:
            known_data = {"names": [], "encodings": []}
            
        known_data["names"].append(name_entry)
        known_data["encodings"].append(encoded_face)
        
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(known_data, f)
        return True
    return False

def take_attendance(subject):
    """Executes facial recognition with Spatial Downscaling and Liveness anti-spoofing."""
    if not os.path.exists(ENCODINGS_FILE):
        st.error("Database Empty: No 128-d embeddings found. Enroll students first.")
        return

    with open(ENCODINGS_FILE, 'rb') as f:
        known_data = pickle.load(f)
        
    known_names = known_data["names"]
    known_encodings = known_data["encodings"]

    video = cv2.VideoCapture(0)
    st.warning(f"Session: {subject}. YOU MUST BLINK to verify liveness before logging. Press 'q' to quit.")
    
    EAR_THRESHOLD = 0.22  
    liveness_verified = False
    process_this_frame = True
    
    face_locations = []
    face_encodings = []
    face_landmarks_list = []

    while True:
        ret, frame = video.read()
        if not ret: continue
        
        # CPU OPTIMIZATION: Shrink image to 50% to reduce mathematical processing load by 75%
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Enforce memory contiguity and 8-bit casting
        rgb_small_frame = np.ascontiguousarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)).astype(np.uint8)
        
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
            
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            
            # Scale coordinates back up by 2 to match the original display frame
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # LIVENESS DETECTION
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Scale landmark points up by 2 for the UI overlay
            for point in left_eye + right_eye:
                pt = (point[0] * 2, point[1] * 2)
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)
                
            if ear < EAR_THRESHOLD:
                liveness_verified = True 
                
            # RECOGNITION LOGIC
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown Face"
            
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            
            # UI Rendering
            box_color = (0, 255, 0) if liveness_verified else (0, 0, 255)
            status_text = "LIVENESS VERIFIED" if liveness_verified else "SPOOF DETECTED: BLINK!"
            
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # ATTENDANCE LOGGING (Listen globally for keys)
            key = cv2.waitKey(1)
            if key == ord('o'):
                if not liveness_verified:
                    st.toast("Spoof Prevention: Cannot log attendance. No blink detected.")
                elif name == "Unknown Face":
                    st.toast("Error: Unrecognized biometric profile.")
                else:
                    ts = datetime.now().strftime("%H:%M:%S")
                    date_str = datetime.now().strftime("%d-%m-%Y")
                    day_str = datetime.now().strftime("%A")
                    file_path = f"{ATTENDANCE_DIR}Attendance_{subject}_{date_str}.csv"
                    file_exists = os.path.isfile(file_path)
                    
                    already_marked = False
                    if file_exists:
                        with open(file_path, "r") as f:
                            reader = csv.reader(f)
                            for row in reader:
                                if row and row[0] == name:
                                    already_marked = True
                                    break
                    if already_marked:
                        st.toast(f"⚠️ Conflict: {name} already logged.")
                    else:
                        with open(file_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            if not file_exists: writer.writerow(['NAME', 'DATE', 'DAY', 'TIME', 'SUBJECT'])
                            writer.writerow([name, date_str, day_str, ts, subject])
                        st.toast(f"✅ Secure Success: Logged {name}.")
                        liveness_verified = False 

        cv2.imshow("Attendance by Face Track AI", frame)
        if cv2.waitKey(1) == ord('q'): break

    video.release()
    cv2.destroyAllWindows()

# --- ENHANCED AUTHENTICATION GATEWAY ---
if not st.session_state['logged_in']:
    st.markdown("""
        <style>
        [data-testid="collapsedControl"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
        .stApp { background-color: #f4f5f7; }
        
        .dark-panel {
            background-color: #1e1e1e;
            color: white; 
            padding: 50px; 
            border-radius: 20px;
            height: 100%; 
            min-height: 500px; 
            display: flex; 
            flex-direction: column; 
            justify-content: start;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .dark-panel h1 { font-size: 2.8rem; font-weight: bold; line-height: 1.2; margin-bottom: 5px; color: white;}
        .highlight-text { color: #38bdf8; }
        .dark-panel p { color: #e5e7eb; font-size: 1.2rem; margin-top: 10px; font-weight: 500; }
        
        div[data-baseweb="input"] {
            border-radius: 8px !important;
            background-color: #f3f4f6 !important;
            border: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4, 1, 4])
    with col1:
        st.markdown("""
            <div class="dark-panel">
                <h1>Attendance <span style="font-size: 0.6em; font-weight: 400;">desk</span><br>
                <span class="highlight-text">AI Face Recognition</span></h1>
                <p>AI-Driven Attendance for Smarter Campuses.</p>
                <br><br>
                <div style="display: flex; align-items: center; gap: 15px; margin-top: auto;">
                    <div style="font-size: 2.5rem; color: #38bdf8;">👤</div>
                    <div style="font-size: 0.9rem; color: #9ca3af; line-height: 1.4;">
                        Secure biometric tracking<br>and automated analytics.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.write(""); st.write(""); st.write("")
        st.header("Sign In")
        user_in = st.text_input("Username", placeholder="Enter your admin username")
        pass_in = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if st.button("Sign In", type="primary", use_container_width=True):
            if os.path.exists(CRED_FILE):
                df = pd.read_csv(CRED_FILE, dtype=str)
                record = df[(df['Username'].str.strip() == user_in.strip()) & (df['Password'].str.strip() == pass_in.strip())]
                if not record.empty:
                    st.session_state['logged_in'] = True
                    st.session_state['admin_name'] = record.iloc[0]['Full Name']
                    st.session_state['page'] = "Home"
                    st.rerun()
                else: st.error("Access Denied: Invalid parameters.")
            else: st.error("Database Error: No admin records found.")
        st.divider()
        st.caption("System Administrator Access Only.")

else:
    # --- AUTHENTICATED SIDEBAR & GENERAL CSS ---
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { background-color: #435b5a !important; }
        [data-testid="stSidebar"] * { color: white !important; }
        .stApp { background-color: #f0f2f6; }
        
        .sidebar-welcome-card {
            background-color: #3f5151; padding: 15px; border-radius: 10px;
            color: white; margin-bottom: 20px;
        }
        
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background-color: transparent !important; border: 1px solid white !important;
            border-radius: 5px !important; text-align: left !important; justify-content: flex-start !important;
            color: white !important; font-weight: normal; margin-bottom: 10px;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background-color: rgba(255,255,255,0.1) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR COMPONENT ---
    st.sidebar.markdown("### System Gateway")
    
    st.sidebar.markdown(f"""
        <div class="sidebar-welcome-card">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 2rem;">👨‍💼</div>
                <div>
                    <div style="font-size: 0.8rem; opacity: 0.8;">Welcome,</div>
                    <div style="font-weight: bold; font-size: 1rem;">{st.session_state['admin_name']}</div>
                </div>
            </div>
            <div style="font-size: 0.8rem; margin-top: 10px; color: #a0a8a8;">Account Settings ⚙️</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("📊 Dashboard", use_container_width=True):
        st.session_state['page'] = "Home"
        st.session_state['selected_log'] = None
        st.rerun()
        
    if st.sidebar.button("Terminate Session (Logout)", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['admin_name'] = "Unknown Admin"
        st.session_state['page'] = "Home"
        st.session_state['selected_log'] = None
        st.rerun()

    # --- ROUTING LOGIC & SPECIFIC PAGE CSS ---
    
    if st.session_state['page'] == "Home":
        st.markdown("""
            <style>
                .dash-card {
                    background-color: #e2e8e8; padding: 20px; border-radius: 10px 10px 0 0;
                    min-height: 140px; border: 1px solid #d0d5d5; border-bottom: none; margin-bottom: -15px;
                }
                .dash-title { font-size: 1.4rem; font-weight: 700; color: black; display: flex; align-items: center; gap: 10px; margin-bottom: 10px;}
                .dash-text { font-size: 0.95rem; color: #111; line-height: 1.4; }
                
                .stMain div[data-testid="stButton"] > button {
                    background-color: #b73229 !important; color: white !important;
                    border-radius: 0 0 10px 10px !important; border: none !important;
                    width: 100%; font-weight: bold; padding: 10px; margin-bottom: 25px;
                }
                .stMain div[data-testid="stButton"] > button:hover { background-color: #8f261f !important; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<h1 style='color: black; margin-bottom: 30px;'>🏛️ Smart Attendance Management System</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<div class="dash-card"><div class="dash-title">👤 Take Attendance</div><div class="dash-text">Activate real-time biometric tracking to record subject attendance.</div></div>', unsafe_allow_html=True)
            if st.button("Open Recognition Portal", use_container_width=True): 
                st.session_state['page'] = "Take Attendance"
                st.rerun()
            
            st.markdown('<div class="dash-card"><div class="dash-title">📈 Attendance Analysis</div><div class="dash-text">Visualize attendance trends, generate reports, and export subject-specific CSV logs.</div></div>', unsafe_allow_html=True)
            if st.button("Open Live Logs", use_container_width=True): 
                st.session_state['page'] = "Live Dashboard"
                st.rerun()

        with col2:
            st.markdown('<div class="dash-card"><div class="dash-title">📝 New Student Register</div><div class="dash-text">Securely enroll new student biometric profiles and link matrix data.</div></div>', unsafe_allow_html=True)
            if st.button("Open Registration", use_container_width=True): 
                st.session_state['page'] = "New Student Registration"
                st.rerun()
            
            st.markdown('<div class="dash-card"><div class="dash-title">🛠️ Admin & Students Registered</div><div class="dash-text">Oversee institutional administrator access and manage the comprehensive student directory.</div></div>', unsafe_allow_html=True)
            if st.button("Open Database Settings", use_container_width=True): 
                st.session_state['page'] = "System Administration"
                st.rerun()

    elif st.session_state['page'] == "Take Attendance":
        
        st.markdown("""
            <style>
                .stMain div[data-testid="stButton"] > button {
                    background-color: #b73229 !important;
                    color: white !important;
                    border-radius: 5px !important;
                    font-weight: bold;
                    height: 2.7rem; 
                    margin-top: 2px;
                }
                .stMain div[data-testid="stButton"] > button:hover { background-color: #8f261f !important; }
                .stSelectbox div[data-baseweb="select"] > div {
                    border-radius: 5px !important;
                }
            </style>
        """, unsafe_allow_html=True)

        st.write("") 
        
        _, center_col, _ = st.columns([1, 2, 1])
        
        with center_col:
            st.markdown("""
                <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); color: black;">
                    <div style="display: flex; align-items: center; margin-bottom: 20px;">
                        <div style="font-size: 3rem; margin-right: 20px;">👤</div>
                        <div>
                            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #111;">Take Attendance</h2>
                            <p style="margin: 5px 0 0; color: #666; font-size: 1rem;">Start the real-time biometric tracking to record subject attendance.</p>
                        </div>
                    </div>
                    <div style="font-weight: bold; font-size: 0.95rem; color: #555; margin-bottom: -15px;">Set Academic Subject</div>
                </div>
            """, unsafe_allow_html=True)

            st.write("") 
            
            input_cols = st.columns([3, 1])
            with input_cols[0]:
                subject_choice = st.selectbox("Subject", ["DSA", "PYTHON", "DBMS", "COA"], label_visibility="collapsed")
            with input_cols[1]:
                engage_btn = st.button("Start", use_container_width=True)

            if engage_btn:
                take_attendance(subject_choice)

    elif st.session_state['page'] == "New Student Registration":
        st.header("📝 Biometric Enrollment")
        st.write("Extracting 128-D matrix directly from live video. Haar Cascades are obsolete.")
        name = st.text_input("Legal Name")
        roll = st.text_input("Roll Number")
        if st.button("Initiate Capture"):
            if name and roll: 
                if capture_new_student(name, roll): st.success("128-D Biometric Matrix compiled and saved.")
            else: st.error("Validation Error: Name and Roll Number missing.")

    elif st.session_state['page'] == "Live Dashboard":
        st.header("📈 Live Subject-Specific Logs")
        if os.path.exists(ATTENDANCE_DIR):
            files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.csv')]
            if files:
                files = sorted(files, reverse=True)
                selected = st.selectbox("Select Target CSV File", files, index=0)
                st.divider()
                if selected:
                    file_path = os.path.join(ATTENDANCE_DIR, selected)
                    try:
                        df_log = pd.read_csv(file_path)
                        st.metric("Total Unique Entries", len(df_log['NAME'].unique()) if 'NAME' in df_log.columns else 0)
                        st.table(df_log) 
                    except Exception as e: 
                        st.error(f"Data Read Error. {e}")
            else: 
                st.info("Database Empty. No logs found.")
        else: 
            st.error("Directory Error.")

    elif st.session_state['page'] == "System Administration":
        st.header("🛠️ Admin & Students Registered")
        st.subheader("📋 Registered Student Biometric Database (128-D Encodings)")
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                known_data = pickle.load(f)
            raw_names = known_data["names"]
            parsed_data = [[e.split("(")[0].strip(), e.split("(")[1].replace(")", "").strip()] if "(" in e else [e, "N/A"] for e in raw_names]
            df_students = pd.DataFrame(parsed_data, columns=["Name", "Roll Number"])
            st.metric("Total Enrolled Students", len(df_students))
            st.table(df_students)
            
            st.markdown("### ⚠️ Purge Biometric Record")
            student_to_delete = st.selectbox("Select Target Profile", raw_names)
            if st.button("Execute Profile Deletion", type="primary"):
                idx = known_data["names"].index(student_to_delete)
                known_data["names"].pop(idx)
                known_data["encodings"].pop(idx)
                with open(ENCODINGS_FILE, 'wb') as f:
                    pickle.dump(known_data, f)
                st.warning(f"Purged {student_to_delete}.")
                st.rerun()
        else: st.info("No biometrics registered.")
        
        st.divider()
        st.subheader("👔 Registered Administrators")
        if os.path.exists(CRED_FILE):
            df_ad = pd.read_csv(CRED_FILE, dtype=str)
            display_df = df_ad[['Full Name', 'Username']].copy()
            st.metric("Total Administrators", len(display_df))
            st.table(display_df)
            
            st.markdown("### ⚠️ Purge Administrator")
            admin_to_delete = st.selectbox("Target Admin Username", df_ad['Username'].unique())
            if st.button("Execute Admin Deletion", type="primary"):
                if len(df_ad) <= 1:
                    st.error("Operation Denied: The system requires at least one active administrator.")
                else:
                    df_ad[df_ad['Username'] != admin_to_delete].to_csv(CRED_FILE, index=False)
                    st.warning(f"Purged Admin: {admin_to_delete}")
                    st.rerun()
        else: st.info("No administrators found.")
        st.divider()
        st.subheader("➕ Provision New Administrator")
        col_fn, col_un, col_pw = st.columns(3)
        with col_fn: fn = st.text_input("Full Name")
        with col_un: un = st.text_input("Username")
        with col_pw: pw = st.text_input("Password", type="password")
        if st.button("Write Admin to Database", type="primary"):
            if fn and un and pw:
                file_exists = os.path.isfile(CRED_FILE)
                with open(CRED_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists: writer.writerow(["Full Name", "Username", "Password"])
                    writer.writerow([fn, un, pw])
                st.success(f"Administrator {fn} provisioned.")
                st.rerun()
            else: st.error("Inputs required.")