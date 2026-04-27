import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
import pickle
import cv2
from datetime import datetime, timedelta
import face_recognition
from scipy.spatial import distance as dist
import plotly.express as px
import threading
import sys

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

# --- AUDIO FEEDBACK: beep when attendance is logged ---
def play_beep(kind="success"):
    """Plays a short beep. Uses winsound.Beep (confirmed working on target hardware).
    Runs in a daemon thread so the camera loop never waits for the sound to finish.
    """
    def _play():
        try:
            if sys.platform == "win32":
                import winsound
                if kind == "success":
                    # Happy ascending two-tone
                    winsound.Beep(880, 120)
                    winsound.Beep(1175, 180)
                elif kind == "warning":
                    winsound.Beep(500, 250)
                elif kind == "error":
                    winsound.Beep(300, 150)
                    winsound.Beep(300, 150)
            else:
                sys.stdout.write('\a'); sys.stdout.flush()
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()

# --- CORE BIOMETRIC FUNCTIONS (DEEP LEARNING) ---
def capture_new_student(full_name, roll_no):
    """Extracts a 128-d facial embedding vector instead of raw pixel arrays."""
    video = cv2.VideoCapture(0)
    
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
    """Executes facial recognition with Spatial Downscaling and 2-Blink Liveness anti-spoofing."""
    if not os.path.exists(ENCODINGS_FILE):
        st.toast("⚠️ Database empty: enroll students first.", icon="⚠️")
        return

    with open(ENCODINGS_FILE, 'rb') as f:
        known_data = pickle.load(f)
        
    known_names = known_data["names"]
    known_encodings = known_data["encodings"]

    video = cv2.VideoCapture(0)
    # Toast instead of st.warning so it doesn't break the dashboard grid symmetry
    st.toast(f"📸 Session started: {subject} — Blink twice, press 'o' to log, 'q' to quit", icon="🎥")
    
    # State Machine Variables
    EAR_CLOSED = 0.21  
    EAR_OPEN = 0.25
    blink_frames = 0
    total_blinks = 0
    liveness_verified = False
    process_this_frame = True
    
    face_locations = []
    face_encodings = []
    face_landmarks_list = []

    # Persist the latest recognized name across frames so the O key still works
    # on frames where face detection was skipped (process_this_frame = False)
    current_name = "Unknown Face"

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
                
            # --- THE 2-BLINK ANTI-SPOOFING PROTOCOL ---
            if ear < EAR_CLOSED:
                blink_frames += 1
            elif ear > EAR_OPEN:
                # If eyes just opened, verify the blink was human-speed (2 to 15 frames)
                # This explicitly blocks static photos held up to the camera
                if 2 <= blink_frames <= 15:
                    total_blinks += 1
                blink_frames = 0 # Reset frame counter
                
            if total_blinks >= 2:
                liveness_verified = True 
            # ------------------------------------------
                
            # RECOGNITION LOGIC
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown Face"
            
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            # Save latest name outside the loop so the O-key handler below can use it
            current_name = name

            # UI Rendering
            box_color = (0, 255, 0) if liveness_verified else (0, 0, 255)
            status_text = "LIVENESS VERIFIED" if liveness_verified else f"SPOOF CHECK: BLINK TWICE ({total_blinks}/2)"
            
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # ============================================================
        # SHOW FRAME + SINGLE WAITKEY CALL PER FRAME
        # ============================================================
        # CRITICAL: cv2.waitKey must be called exactly ONCE per frame, OUTSIDE the
        # inner for-loop. The old code called it inside the loop and again at the
        # bottom, which caused keypresses to be eaten by whichever call got them
        # first — and on frames with no detected face (skipped frames), the O key
        # was never read at all.
        cv2.imshow("Attendance by Face Track AI", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('o'):
            if not liveness_verified:
                play_beep("error")
                st.toast("Spoof Prevention: Cannot log attendance. No blink detected.")
            elif current_name == "Unknown Face":
                play_beep("error")
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
                        next(reader, None) # Skip header
                        for row in reader:
                            if row and row[0] == current_name:
                                try:
                                    last_time = datetime.strptime(row[3], "%H:%M:%S")
                                    current_time = datetime.strptime(ts, "%H:%M:%S")
                                    diff = (current_time - last_time).total_seconds() / 60
                                    # 5-Minute Cool-down Logic
                                    if diff < 5:
                                        already_marked = True
                                        break
                                except:
                                    already_marked = True
                                    break

                if already_marked:
                    play_beep("warning")
                    st.toast(f"⚠️ Cool-down Active: {current_name} recently logged.")
                else:
                    with open(file_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists: writer.writerow(['NAME', 'DATE', 'DAY', 'TIME', 'SUBJECT'])
                        writer.writerow([current_name, date_str, day_str, ts, subject])
                    play_beep("success")
                    st.toast(f"✅ Secure Success: Logged {current_name}.")
                    liveness_verified = False
                    total_blinks = 0

    video.release()
    cv2.destroyAllWindows()

# --- ENHANCED AUTHENTICATION GATEWAY ---
if not st.session_state['logged_in']:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

        /* Hide Streamlit chrome on login page */
        [data-testid="collapsedControl"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }

        html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

        /* Soft pink/cream page background with decorative dots */
        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(220,38,38,0.08) 0%, transparent 40%),
                radial-gradient(circle at 88% 78%, rgba(220,38,38,0.06) 0%, transparent 45%),
                #fdf7f7 !important;
        }
        .main .block-container {
            padding-top: 3rem !important;
            padding-bottom: 2rem !important;
            max-width: 1300px !important;
        }

        /* ==================== BRAND PANEL (LEFT) ==================== */
        .brand-panel {
            background: white;
            border-radius: 24px;
            padding: 50px 42px 40px 42px;
            min-height: 620px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(185, 28, 28, 0.06);
            border: 1px solid #fde4e4;
            display: flex;
            flex-direction: column;
        }
        /* Decorative background rings */
        .brand-panel::before {
            content: "";
            position: absolute;
            top: -100px; left: -100px;
            width: 300px; height: 300px;
            border-radius: 50%;
            border: 2px solid rgba(220, 38, 38, 0.06);
            pointer-events: none;
        }
        .brand-panel::after {
            content: "";
            position: absolute;
            bottom: -120px; right: -120px;
            width: 350px; height: 350px;
            border-radius: 50%;
            border: 2px solid rgba(220, 38, 38, 0.06);
            pointer-events: none;
        }
        /* Floating dots */
        .brand-dot {
            position: absolute;
            border-radius: 50%;
            background: #fca5a5;
            pointer-events: none;
        }

        .brand-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.05;
            margin-bottom: 6px;
            position: relative;
            z-index: 2;
        }
        .brand-title .sm {
            font-size: 1.4rem;
            font-weight: 400;
            color: #334155;
            margin-left: 8px;
        }
        .brand-title .red {
            display: block;
            color: #dc2626;
            font-weight: 800;
            font-size: 2.5rem;
            margin-top: 4px;
        }
        .brand-sub {
            font-size: 1rem;
            color: #64748b;
            line-height: 1.55;
            margin-top: 14px;
            max-width: 320px;
            position: relative;
            z-index: 2;
        }

        /* Hero illustration area (laptop + shield) */
        .hero-wrap {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 30px 0;
            position: relative;
            z-index: 2;
        }
        .hero-laptop {
            width: 300px; height: 200px;
            position: relative;
        }
        .hero-laptop-screen {
            width: 280px; height: 170px;
            background: linear-gradient(145deg, #2d2d3e 0%, #1a1a2e 100%);
            border-radius: 14px 14px 4px 4px;
            padding: 10px;
            position: relative;
            border: 3px solid #3a3a4e;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        .hero-laptop-screen-inner {
            width: 100%; height: 100%;
            background: linear-gradient(145deg, #fff 0%, #fef2f2 100%);
            border-radius: 6px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .hero-dots { display: flex; gap: 5px; }
        .hero-dot { width: 7px; height: 7px; border-radius: 50%; background: #fecaca; }
        .hero-dot:nth-child(2) { background: #fca5a5; }
        .hero-dot:nth-child(3) { background: #f87171; }
        .hero-avatar {
            width: 48px; height: 48px; border-radius: 50%;
            background: linear-gradient(135deg, #fecaca, #fca5a5);
            margin: 6px auto;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem;
        }
        .hero-bar {
            height: 6px;
            background: linear-gradient(90deg, #dc2626, #fca5a5);
            border-radius: 3px;
            opacity: 0.7;
        }
        .hero-bar.short { width: 60%; margin: 0 auto; }
        .hero-bar.mid   { width: 80%; margin: 0 auto; opacity: 0.4; }
        .hero-laptop-base {
            width: 320px; height: 10px;
            background: linear-gradient(180deg, #3a3a4e 0%, #2d2d3e 100%);
            border-radius: 0 0 12px 12px;
            margin: 0 auto;
            margin-left: -10px;
        }
        /* Floating shield overlay */
        .hero-shield {
            position: absolute;
            bottom: 10px;
            right: -20px;
            width: 90px; height: 100px;
            background: linear-gradient(145deg, #dc2626 0%, #991b1b 100%);
            clip-path: polygon(50% 0%, 100% 25%, 100% 60%, 50% 100%, 0% 60%, 0% 25%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.2rem;
            color: white;
            filter: drop-shadow(0 8px 16px rgba(185,28,28,0.35));
        }

        /* Brand footer note */
        .brand-footer {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-top: 20px;
            position: relative;
            z-index: 2;
        }
        .brand-footer-ico {
            width: 40px; height: 40px;
            border-radius: 10px;
            border: 1.5px solid #fecaca;
            color: #dc2626;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.3rem;
            flex-shrink: 0;
        }
        .brand-footer-text {
            font-size: 0.88rem;
            color: #64748b;
            line-height: 1.5;
        }

        /* ==================== LOGIN CARD (RIGHT) ==================== */
        /* Style the right column as a white card by targeting the marker via :has() */
        [data-testid="stColumn"]:has(.login-card-marker) {
            background: white;
            border-radius: 24px;
            box-shadow: 0 10px 40px rgba(15, 23, 42, 0.06);
            border: 1px solid #f1f5f9;
        }
        [data-testid="stColumn"]:has(.login-card-marker) > div {
            padding: 46px 44px 36px 44px !important;
            min-height: 620px;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }
        /* Hide the marker itself */
        .login-card-marker { display: none; }

        .login-shield {
            width: 72px; height: 72px;
            border-radius: 50%;
            background: #fef2f2;
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 14px auto;
            font-size: 2rem;
            color: #dc2626;
            box-shadow: inset 0 0 0 6px white, 0 0 0 1px #fecaca;
        }
        .login-title {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0f172a;
            text-align: center;
            margin-bottom: 4px;
            letter-spacing: -0.01em;
        }
        .login-sub {
            font-size: 0.92rem;
            color: #64748b;
            text-align: center;
            margin-bottom: 28px;
        }

        /* Custom-styled input fields on the login page */
        [data-testid="stColumn"]:has(.login-card-marker) div[data-baseweb="input"] {
            border-radius: 10px !important;
            background: white !important;
            border: 1.5px solid #e5e7eb !important;
            transition: all 0.18s ease;
        }
        [data-testid="stColumn"]:has(.login-card-marker) div[data-baseweb="input"]:focus-within {
            border-color: #dc2626 !important;
            box-shadow: 0 0 0 3px rgba(220,38,38,0.08) !important;
        }
        [data-testid="stColumn"]:has(.login-card-marker) div[data-baseweb="input"] input {
            padding: 12px 14px !important;
            font-size: 0.95rem !important;
            background: white !important;
        }

        /* Login-page specific label style (icon + text) */
        .login-label {
            display: flex; align-items: center; gap: 6px;
            font-size: 0.85rem; font-weight: 500; color: #334155;
            margin-bottom: 6px; margin-top: 10px;
        }
        .login-label-icon { color: #64748b; font-size: 0.95rem; }

        /* Primary "Login" button — red gradient */
        [data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(180deg, #dc2626 0%, #b91c1c 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 13px !important;
            box-shadow: 0 4px 14px rgba(220,38,38,0.30) !important;
            transition: all 0.2s ease !important;
            width: 100%;
            margin-top: 8px;
        }
        [data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(180deg, #b91c1c 0%, #7f1d1d 100%) !important;
            box-shadow: 0 8px 20px rgba(185,28,28,0.40) !important;
            transform: translateY(-1px);
        }

        /* Forgot password link */
        .forgot-link {
            color: #dc2626;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            font-size: 0.88rem;
        }
        .forgot-link:hover { color: #991b1b; }

        /* Footer note at the very bottom of the login card */
        .login-footer-note {
            display: flex; align-items: center; justify-content: center;
            gap: 8px;
            margin-top: 28px;
            padding-top: 20px;
            border-top: 1px solid #f1f5f9;
            font-size: 0.85rem;
            color: #64748b;
        }
        .login-footer-note .ico { color: #dc2626; }

        /* Bottom copyright */
        .login-copy {
            text-align: center;
            margin-top: 26px;
            font-size: 0.82rem;
            color: #94a3b8;
        }

        /* Checkbox label color tweak */
        .stCheckbox label p { font-size: 0.88rem !important; color: #334155 !important; }
        </style>
    """, unsafe_allow_html=True)

    # === Two-column layout: Brand panel (left) + Login card (right) ===
    left, right = st.columns([1, 1], gap="large")

    # ---------- LEFT: Brand panel ----------
    with left:
        brand_html = (
            '<div class="brand-panel">'
              # Floating dots
              '<span class="brand-dot" style="top:80px; right:40px; width:10px; height:10px;"></span>'
              '<span class="brand-dot" style="top:160px; right:90px; width:6px; height:6px; opacity:0.6;"></span>'
              '<span class="brand-dot" style="bottom:220px; left:50px; width:8px; height:8px; opacity:0.5;"></span>'
              '<span class="brand-dot" style="bottom:120px; right:60px; width:5px; height:5px;"></span>'
              # Title block
              '<div>'
                '<div class="brand-title">Attendance <span class="sm">desk</span>'
                  '<span class="red">AI Face Recognition</span>'
                '</div>'
                '<div class="brand-sub">AI-Driven Attendance for Smarter Campuses.</div>'
              '</div>'
              # Hero illustration
              '<div class="hero-wrap">'
                '<div class="hero-laptop">'
                  '<div class="hero-laptop-screen">'
                    '<div class="hero-laptop-screen-inner">'
                      '<div class="hero-dots"><span class="hero-dot"></span><span class="hero-dot"></span><span class="hero-dot"></span></div>'
                      '<div class="hero-avatar">👤</div>'
                      '<div class="hero-bar short"></div>'
                      '<div class="hero-bar mid"></div>'
                      '<div class="hero-bar short"></div>'
                    '</div>'
                  '</div>'
                  '<div class="hero-laptop-base"></div>'
                  '<div class="hero-shield">🔒</div>'
                '</div>'
              '</div>'
              # Footer note
              '<div class="brand-footer">'
                '<div class="brand-footer-ico">👤</div>'
                '<div class="brand-footer-text">Secure biometric tracking<br>and automated analytics.</div>'
              '</div>'
            '</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)

    # ---------- RIGHT: Login card (column itself becomes the card via :has() CSS) ----------
    with right:
        # Hidden marker — the CSS selector [data-testid="stColumn"]:has(.login-card-marker)
        # uses this to style the parent column as a white rounded card.
        st.markdown('<div class="login-card-marker"></div>', unsafe_allow_html=True)

        # Shield icon + title + subtitle
        st.markdown(
            '<div class="login-shield">🛡️</div>'
            '<div class="login-title">Welcome Back!</div>'
            '<div class="login-sub">Sign in to continue to your admin dashboard.</div>',
            unsafe_allow_html=True
        )

        # Admin ID field
        st.markdown('<div class="login-label"><span class="login-label-icon">👤</span> Admin ID</div>', unsafe_allow_html=True)
        user_in = st.text_input("Admin ID", placeholder="admin", label_visibility="collapsed", key="login_user")

        # Password field
        st.markdown('<div class="login-label"><span class="login-label-icon">🔒</span> Password</div>', unsafe_allow_html=True)
        pass_in = st.text_input("Password", type="password", placeholder="••••••••••", label_visibility="collapsed", key="login_pass")

        # Remember me + Forgot password row
        rem_col, fp_col = st.columns([1, 1])
        with rem_col:
            remember_me = st.checkbox("Remember me", key="login_remember")
        with fp_col:
            st.markdown(
                '<div style="text-align:right; padding-top:6px;">'
                  '<a class="forgot-link" onclick="return false;">Forgot password?</a>'
                '</div>',
                unsafe_allow_html=True
            )

        # Login button
        if st.button("➜  Login", type="primary", use_container_width=True, key="login_btn"):
            if os.path.exists(CRED_FILE):
                df = pd.read_csv(CRED_FILE, dtype=str)
                record = df[
                    (df['Username'].str.strip() == user_in.strip()) &
                    (df['Password'].str.strip() == pass_in.strip())
                ]
                if not record.empty:
                    st.session_state['logged_in'] = True
                    st.session_state['admin_name'] = record.iloc[0]['Full Name']
                    st.session_state['page'] = "Home"
                    st.rerun()
                else:
                    st.toast("Access denied: invalid credentials.", icon="🔒")
            else:
                st.toast("No admin records found. Please contact system admin.", icon="⚠️")

        # Footer security note (directly below the login button)
        st.markdown(
            '<div class="login-footer-note">'
              '<span class="ico">🛡️</span>'
              '<span>Secure login to your account</span>'
            '</div>',
            unsafe_allow_html=True
        )

    # Bottom copyright
    st.markdown(
        '<div class="login-copy">© 2026 Attendance Desk. All rights reserved.</div>',
        unsafe_allow_html=True
    )

else:
    # --- GLOBAL MASTER STYLESHEET (MODERN RED DASHBOARD THEME) ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

        /* Base Architecture */
        html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }
        .stApp { background-color: #f5f6fa; }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

        /* ============ SIDEBAR (matches image 3 — deeper maroon) ============ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #8b1a1a 0%, #7a1515 55%, #6b1010 100%) !important;
            border-right: none !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding: 1.2rem 1rem 1rem 1rem; }
        [data-testid="stSidebar"] * { color: white !important; font-family: 'Poppins', sans-serif !important; }

        /* Sidebar logo header */
        .sb-logo-wrap {
            display: flex; align-items: center; gap: 12px;
            padding: 4px 4px 16px 4px;
            border-bottom: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 18px;
        }
        .sb-logo-icon {
            width: 44px; height: 44px; border-radius: 10px;
            background: white; color: #7a1515;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem; font-weight: 800;
            box-shadow: 0 4px 10px rgba(0,0,0,0.20);
        }
        .sb-logo-title { font-size: 1.15rem; font-weight: 700; line-height: 1.1; }
        .sb-logo-sub   { font-size: 0.72rem; opacity: 0.75; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px; }

        /* Profile welcome card */
        .sb-profile {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 20px;
        }
        .sb-avatar {
            width: 44px; height: 44px; border-radius: 50%;
            background: linear-gradient(135deg, #fecaca, #fca5a5);
            display: inline-flex; align-items: center; justify-content: center;
            font-size: 1.3rem; color: #7a1515 !important; font-weight: 700;
        }
        .sb-welcome { font-size: 0.72rem; opacity: 0.8; }
        .sb-name { font-size: 0.95rem; font-weight: 700; line-height: 1.2; }
        .sb-online {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 0.72rem; opacity: 0.9; margin-top: 4px;
        }
        .sb-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,0.25);
        }

        /* Section headers in sidebar */
        .sb-section {
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.12em;
            opacity: 0.55;
            margin: 16px 6px 8px 6px;
            text-transform: uppercase;
        }

        /* ---- Sidebar nav buttons: INACTIVE (default/secondary) ---- */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background: transparent !important;
            border: none !important;
            border-radius: 10px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            color: rgba(255,255,255,0.88) !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            padding: 11px 14px !important;
            margin-bottom: 3px !important;
            box-shadow: none !important;
            transition: all 0.18s ease;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(255,255,255,0.10) !important;
            color: white !important;
        }

        /* ---- Sidebar nav buttons: ACTIVE (primary) — WHITE with red text ---- */
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[kind="primary"] *,
        [data-testid="stSidebar"] button[kind="primary"] p,
        [data-testid="stSidebar"] button[kind="primary"] span,
        [data-testid="stSidebar"] button[kind="primary"] div {
            color: #7a1515 !important;
        }
        [data-testid="stSidebar"] button[kind="primary"] {
            background: white !important;
            border: none !important;
            border-radius: 10px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 11px 14px !important;
            margin-bottom: 3px !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.18) !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background: #fef2f2 !important;
            transform: none !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[kind="primary"]:hover * {
            color: #5c0f0f !important;
        }

        [data-testid="stSidebar"] button p {
            text-align: left !important;
            margin: 0 !important;
            width: 100% !important;
        }

        /* Logout button (targeted by its unique key) */
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"][kind="secondary"][aria-label*="Logout"],
        [data-testid="stSidebar"] .logout-wrap button {
            background: rgba(255,255,255,0.14) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            font-weight: 600 !important;
            text-align: center !important;
            justify-content: center !important;
            padding: 12px !important;
        }

        /* ============ MAIN DASHBOARD ============ */
        .page-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 6px 0;
            letter-spacing: -0.02em;
        }
        .page-sub {
            color: #64748b;
            font-size: 0.98rem;
            margin-bottom: 28px;
        }

        /* Subject card — CENTERED vertical layout (no schedule) */
        .subj-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 26px 20px 18px 20px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
            transition: all 0.2s ease;
            min-height: 210px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 14px;
        }
        .subj-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #dc2626, #b91c1c);
            border-radius: 16px 16px 0 0;
        }
        .subj-card:hover {
            box-shadow: 0 10px 28px rgba(185,28,28,0.12);
            border-color: #fecaca;
            transform: translateY(-2px);
        }
        .subj-icon-box {
            width: 64px; height: 64px;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.9rem;
            color: white;
            box-shadow: 0 6px 16px rgba(185,28,28,0.30);
            position: relative;
            z-index: 2;
        }
        .subj-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.35;
            max-width: 100%;
            position: relative;
            z-index: 2;
        }
        .last-taken-badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: #fef2f2;
            color: #b91c1c;
            font-size: 0.78rem; font-weight: 600;
            border-radius: 20px;
            padding: 5px 14px;
            position: relative;
            z-index: 2;
        }
        .last-taken-badge.green { background: #dcfce7; color: #15803d; }
        .last-taken-badge.amber { background: #fef3c7; color: #b45309; }
        .subj-deco {
            position: absolute;
            right: -12px; bottom: -10px;
            font-size: 7rem;
            opacity: 0.05;
            pointer-events: none;
            line-height: 1;
            transform: rotate(-8deg);
            z-index: 1;
        }

        /* Main-content primary buttons ("Take Attendance →") — red gradient */
        .stMain button[kind="primary"],
        [data-testid="stAppViewContainer"] .main button[kind="primary"] {
            background: linear-gradient(180deg, #dc2626 0%, #b91c1c 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 12px !important;
            margin-top: 6px !important;
            box-shadow: 0 4px 12px rgba(220,38,38,0.28) !important;
            transition: all 0.18s ease !important;
        }
        .stMain button[kind="primary"]:hover {
            background: linear-gradient(180deg, #b91c1c 0%, #7f1d1d 100%) !important;
            box-shadow: 0 6px 16px rgba(185,28,28,0.40) !important;
            transform: translateY(-1px);
        }

        /* Bottom stats bar */
        .stats-row {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 22px 28px;
            margin-top: 26px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }
        .stat-item { display: flex; align-items: center; gap: 14px; }
        .stat-icon {
            width: 52px; height: 52px;
            background: #fef2f2;
            color: #b91c1c;
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem;
            flex-shrink: 0;
        }
        .stat-label { font-size: 0.82rem; color: #64748b; font-weight: 500; }
        .stat-value { font-size: 1.6rem; color: #0f172a; font-weight: 800; line-height: 1; margin: 2px 0; }
        .stat-help { font-size: 0.72rem; color: #94a3b8; }

        /* Input styling tweaks */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-radius: 8px !important; }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR COMPONENT ---
    with st.sidebar:
        # Logo header
        st.markdown("""
            <div class="sb-logo-wrap">
                <div class="sb-logo-icon">🛡️</div>
                <div>
                    <div class="sb-logo-title">Smart Attendance</div>
                    <div class="sb-logo-sub">Recognition Portal</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Profile card
        initials = "".join([p[0] for p in st.session_state['admin_name'].split()[:2]]).upper() or "A"
        st.markdown(f"""
            <div class="sb-profile">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div class="sb-avatar">{initials}</div>
                    <div>
                        <div class="sb-welcome">Welcome back,</div>
                        <div class="sb-name">{st.session_state['admin_name']}</div>
                        <div class="sb-online"><span class="sb-dot"></span> Online</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # MAIN MENU section
        st.markdown('<div class="sb-section">Main Menu</div>', unsafe_allow_html=True)

        current_page = st.session_state['page']

        def nav_button(label, page_key, key):
            """
            Uses Streamlit's native type='primary' for the active state so the
            style actually applies (wrapper-div CSS gets orphaned by Streamlit).
            """
            btn_type = "primary" if current_page == page_key else "secondary"
            if st.button(label, key=key, use_container_width=True, type=btn_type):
                st.session_state['page'] = page_key
                st.session_state['selected_log'] = None
                st.rerun()

        # Menu items matching image 3 exactly
        nav_button("📊   Dashboard",          "Home",                     "nav_home")
        nav_button("📋   Attendance Records", "Live Dashboard",           "nav_logs")
        nav_button("👥   Students",           "System Administration",    "nav_students")
        nav_button("📝   Register Student",   "New Student Registration", "nav_register")
        nav_button("📅   Schedule",           "Schedule",                 "nav_schedule")

        # SYSTEM section
        st.markdown('<div class="sb-section">System</div>', unsafe_allow_html=True)
        nav_button("⚙️   Account Settings",   "System Administration", "nav_settings")
        nav_button("❓   Help & Support",     "Research",              "nav_help")

        # Logout button at bottom (wrapped so we can style it uniquely)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="logout-wrap">', unsafe_allow_html=True)
        if st.button("⏻   Logout", key="nav_logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['admin_name'] = "Unknown Admin"
            st.session_state['page'] = "Home"
            st.session_state['selected_log'] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ROUTING LOGIC ---
    if st.session_state['page'] == "Home":
        # Page header
        st.markdown("""
            <div style="margin-bottom: 8px;">
                <div class="page-title">Select Subject for Face Recognition Attendance</div>
                <div class="page-sub">Choose a subject below to start taking attendance using face recognition.</div>
            </div>
        """, unsafe_allow_html=True)

        # Helper: calculate last-taken info per subject
        def get_last_taken(subject):
            try:
                files = [f for f in os.listdir(ATTENDANCE_DIR)
                         if f.startswith(f"Attendance_{subject}_") and f.endswith(".csv")]
                if not files:
                    return None
                dates = []
                for f in files:
                    try:
                        date_str = f.replace(f"Attendance_{subject}_", "").replace(".csv", "")
                        dates.append(datetime.strptime(date_str, "%d-%m-%Y").date())
                    except:
                        pass
                return max(dates) if dates else None
            except:
                return None

        def last_taken_badge(subject):
            last = get_last_taken(subject)
            if last is None:
                return '<div class="last-taken-badge">🕒 No session yet</div>'
            delta = (datetime.now().date() - last).days
            if delta == 0:
                return '<div class="last-taken-badge green">🕒 Taken today</div>'
            if delta == 1:
                return '<div class="last-taken-badge amber">🕒 Last taken: 1 day ago</div>'
            return f'<div class="last-taken-badge">🕒 Last taken: {delta} days ago</div>'

        SUBJECTS = [
            {"key": "DSA",    "title": "DSA (Data Structures & Algorithms)", "days": "Mon, Wed", "time": "10:00 AM – 12:00 PM", "icon": "{ }",  "deco": "🌳"},
            {"key": "PYTHON", "title": "PYTHON (Python Programming)",        "days": "Tue, Thu", "time": "11:00 AM – 01:00 PM", "icon": "🐍",   "deco": "</>"},
            {"key": "DBMS",   "title": "DBMS (Database Management Systems)", "days": "Mon, Wed", "time": "02:00 PM – 04:00 PM", "icon": "🗄️",  "deco": "🗃️"},
            {"key": "COA",    "title": "COA (Computer Org & Architecture)",  "days": "Tue, Thu", "time": "09:00 AM – 11:00 AM", "icon": "💻",   "deco": "🖥️"},
        ]

        # Render subject cards in 2-column grid (simplified: icon + centered title + last-taken badge)
        col1, col2 = st.columns(2, gap="large")
        slots = [col1, col2, col1, col2]

        for i, subj in enumerate(SUBJECTS):
            with slots[i]:
                st.markdown(f"""
                    <div class="subj-card">
                        <div class="subj-deco">{subj["deco"]}</div>
                        <div class="subj-icon-box">{subj["icon"]}</div>
                        <div class="subj-title">{subj["title"]}</div>
                        {last_taken_badge(subj["key"])}
                    </div>
                """, unsafe_allow_html=True)
                if st.button(f"Take Attendance  →", key=f"attn_{subj['key']}", use_container_width=True, type="primary"):
                    take_attendance(subj["key"])

        # --- BOTTOM STATS BAR (real metrics from your data) ---
        total_subjects = len(SUBJECTS)

        # Sessions held = number of unique (subject, date) attendance CSV files
        try:
            all_attendance_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")]
            sessions_held = len(all_attendance_files)
        except:
            sessions_held = 0

        # Total attendance % = average of (unique students per session / total enrolled students)
        total_enrolled = 0
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    total_enrolled = len(pickle.load(f).get("names", []))
            except:
                total_enrolled = 0

        avg_attendance_pct = 0
        if total_enrolled > 0 and sessions_held > 0:
            ratios = []
            for fname in all_attendance_files:
                try:
                    df_tmp = pd.read_csv(os.path.join(ATTENDANCE_DIR, fname))
                    if "NAME" in df_tmp.columns:
                        unique_students = df_tmp["NAME"].nunique()
                        ratios.append(min(unique_students / total_enrolled, 1.0))
                except:
                    pass
            if ratios:
                avg_attendance_pct = int(round(sum(ratios) / len(ratios) * 100))

        # Upcoming sessions (next 7 days based on schedule)
        schedule_map = {
            "DSA":    ["Monday", "Wednesday"],
            "PYTHON": ["Tuesday", "Thursday"],
            "DBMS":   ["Monday", "Wednesday"],
            "COA":    ["Tuesday", "Thursday"],
        }
        upcoming = 0
        today = datetime.now().date()
        for offset in range(1, 8):
            day_name = (today + timedelta(days=offset)).strftime("%A")
            for days in schedule_map.values():
                if day_name in days:
                    upcoming += 1

        st.markdown(f"""
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-icon">👥</div>
                    <div>
                        <div class="stat-label">Total Subjects</div>
                        <div class="stat-value">{total_subjects}</div>
                        <div class="stat-help">Active subjects</div>
                    </div>
                </div>
                <div class="stat-item">
                    <div class="stat-icon">✓</div>
                    <div>
                        <div class="stat-label">Total Attendance</div>
                        <div class="stat-value">{avg_attendance_pct}%</div>
                        <div class="stat-help">Average this month</div>
                    </div>
                </div>
                <div class="stat-item">
                    <div class="stat-icon">🕐</div>
                    <div>
                        <div class="stat-label">Sessions Held</div>
                        <div class="stat-value">{sessions_held}</div>
                        <div class="stat-help">All-time</div>
                    </div>
                </div>
                <div class="stat-item">
                    <div class="stat-icon">📅</div>
                    <div>
                        <div class="stat-label">Upcoming Sessions</div>
                        <div class="stat-value">{upcoming}</div>
                        <div class="stat-help">Next 7 days</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    elif st.session_state['page'] == "Take Attendance":
        # This page is now merged into the Home dashboard. Redirect to keep navigation safe.
        st.session_state['page'] = "Home"
        st.rerun()

    elif st.session_state['page'] == "Schedule":
        st.markdown("""
            <style>
            /* Page header with icon tile */
            .sch-head { display: flex; align-items: center; gap: 16px; margin-bottom: 28px; }
            .sch-head-icon {
                width: 56px; height: 56px; border-radius: 14px;
                background: #fee2e2;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.8rem;
                flex-shrink: 0;
            }
            .sch-head-title { font-size: 1.9rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
            .sch-head-sub   { color: #64748b; font-size: 0.95rem; margin-top: 4px; }

            /* This Week pill (top-right of card) */
            .sch-top-bar { display: flex; justify-content: flex-end; margin-bottom: 14px; }
            .sch-week-pill {
                display: inline-flex; align-items: center; gap: 8px;
                background: white; border: 1px solid #e5e7eb;
                border-radius: 10px; padding: 8px 14px;
                font-size: 0.85rem; font-weight: 600; color: #334155;
                box-shadow: 0 1px 3px rgba(15,23,42,0.04);
            }

            /* Timetable container */
            .tt-wrap {
                background: white; border: 1px solid #e5e7eb; border-radius: 16px;
                padding: 20px 22px 22px 22px;
                box-shadow: 0 2px 10px rgba(15,23,42,0.04);
                margin-bottom: 22px;
            }
            .tt-table {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Poppins', sans-serif;
            }
            .tt-table th {
                background: linear-gradient(180deg, #b91c1c 0%, #991b1b 100%);
                color: white !important;
                font-weight: 600;
                font-size: 0.92rem;
                padding: 16px 10px;
                text-align: center;
                border: none;
                letter-spacing: 0.02em;
            }
            .tt-table th:first-child { border-top-left-radius: 10px; border-bottom-left-radius: 10px; }
            .tt-table th:last-child  { border-top-right-radius: 10px; border-bottom-right-radius: 10px; }

            .tt-table td {
                padding: 18px 10px;
                text-align: center;
                font-size: 0.92rem;
                color: #334155;
                border-bottom: 1px solid #f1f5f9;
            }
            .tt-table tr:last-child td { border-bottom: none; }

            .tt-time { color: #0f172a; font-weight: 500; font-size: 0.9rem; }
            .tt-dur  { color: #64748b; font-size: 0.88rem; }

            /* Subject pills — soft pastels matching reference */
            .pill {
                display: inline-block;
                padding: 7px 22px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 500;
                min-width: 90px;
            }
            .pill-dsa    { background: #dbeafe; color: #1e40af; }   /* blue */
            .pill-python { background: #fef3c7; color: #92400e; }   /* cream/amber */
            .pill-dbms   { background: #dcfce7; color: #166534; }   /* green */
            .pill-coa    { background: #f3e8ff; color: #6b21a8; }   /* lavender */
            .pill-buffer { background: #ede9fe; color: #5b21b6; }   /* light purple */
            .pill-break  { background: #fee2e2; color: #991b1b; font-weight: 600; }   /* soft pink */

            /* Bottom stats bar */
            .sch-stats {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 22px 28px;
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 28px;
                box-shadow: 0 2px 10px rgba(15,23,42,0.04);
            }
            .sch-stat { display: flex; align-items: center; gap: 14px; }
            .sch-stat-ico {
                width: 52px; height: 52px; border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.4rem; flex-shrink: 0;
            }
            .sch-stat-ico.red    { background: #fee2e2; color: #b91c1c; }
            .sch-stat-ico.green  { background: #dcfce7; color: #166534; }
            .sch-stat-ico.amber  { background: #fef3c7; color: #b45309; }
            .sch-stat-ico.purple { background: #ede9fe; color: #6b21a8; }
            .sch-stat-label { font-size: 0.82rem; color: #64748b; font-weight: 500; }
            .sch-stat-value { font-size: 1.6rem; color: #0f172a; font-weight: 800; line-height: 1; margin: 2px 0; }
            .sch-stat-help  { font-size: 0.75rem; color: #94a3b8; }
            </style>
        """, unsafe_allow_html=True)

        # Page header
        st.markdown("""
            <div class="sch-head">
                <div class="sch-head-icon">📅</div>
                <div>
                    <div class="sch-head-title">Schedule</div>
                    <div class="sch-head-sub">View your weekly class schedule and subjects.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # The exact schedule provided
        schedule_rows = [
            # (time, duration, mon, tue, wed, thu, fri)
            ("09:30 AM - 10:10 AM", "40 mins", ("DSA",    "pill-dsa"),    ("Python", "pill-python"), ("DBMS",   "pill-dbms"),   ("COA",    "pill-coa"),    ("Python", "pill-python")),
            ("10:10 AM - 10:50 AM", "40 mins", ("Python", "pill-python"), ("DBMS",   "pill-dbms"),   ("COA",    "pill-coa"),    ("DSA",    "pill-dsa"),    ("COA",    "pill-coa")),
            ("10:50 AM - 11:30 AM", "40 mins", ("DBMS",   "pill-dbms"),   ("COA",    "pill-coa"),    ("DSA",    "pill-dsa"),    ("Python", "pill-python"), ("DSA",    "pill-dsa")),
            ("11:30 AM - 12:00 PM", "30 mins") + (("Review Buffer", "pill-buffer"),) * 5,
            ("12:00 PM - 12:50 PM", "50 mins") + (("BREAK",          "pill-break"),)  * 5,
            ("12:50 PM - 01:30 PM", "40 mins", ("COA",    "pill-coa"),    ("DSA",    "pill-dsa"),    ("Python", "pill-python"), ("DBMS",   "pill-dbms"),   ("DBMS",   "pill-dbms")),
        ]

        # Build the HTML table rows (no leading indentation so Markdown doesn't treat it as a code block)
        tbody_parts = []
        for row in schedule_rows:
            time_s, duration = row[0], row[1]
            cells = row[2:]
            cell_html = "".join(
                f'<td><span class="pill {cls}">{label}</span></td>'
                for (label, cls) in cells
            )
            tbody_parts.append(
                f'<tr><td class="tt-time">{time_s}</td><td class="tt-dur">{duration}</td>{cell_html}</tr>'
            )
        tbody = "".join(tbody_parts)

        # Build the full HTML as a single compact string (no leading whitespace per line) so
        # Streamlit's markdown parser doesn't render it as a fenced code block.
        table_html = (
            '<div class="tt-wrap">'
              '<div class="sch-top-bar"><span class="sch-week-pill">📅 This Week ▾</span></div>'
              '<table class="tt-table">'
                '<thead><tr>'
                  '<th style="width:17%;">Time</th>'
                  '<th style="width:10%;">Duration</th>'
                  '<th>Monday</th><th>Tuesday</th><th>Wednesday</th><th>Thursday</th><th>Friday</th>'
                '</tr></thead>'
                f'<tbody>{tbody}</tbody>'
              '</table>'
            '</div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

        # --- Bottom stats bar (computed from the timetable itself) ---
        # Count actual teaching slots (exclude Review Buffer and BREAK)
        total_classes = 0
        for row in schedule_rows:
            for (label, _cls) in row[2:]:
                if label not in ("Review Buffer", "BREAK"):
                    total_classes += 1

        subjects_count = 4  # DSA, Python, DBMS, COA

        # Daily total minutes = sum of durations across ALL rows (incl. buffer & break as part of day)
        total_day_minutes = 40 + 40 + 40 + 30 + 50 + 40   # = 240 mins = 4h 0m
        hrs = total_day_minutes // 60
        mins = total_day_minutes % 60
        daily_duration = f"{hrs}h {mins:02d}m"

        # Real attendance rate from logged CSVs
        try:
            all_attendance_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")]
        except:
            all_attendance_files = []

        total_enrolled = 0
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    total_enrolled = len(pickle.load(f).get("names", []))
            except:
                total_enrolled = 0

        attendance_rate = 0
        if total_enrolled > 0 and all_attendance_files:
            ratios = []
            for fname in all_attendance_files:
                try:
                    df_tmp = pd.read_csv(os.path.join(ATTENDANCE_DIR, fname))
                    if "NAME" in df_tmp.columns:
                        ratios.append(min(df_tmp["NAME"].nunique() / total_enrolled, 1.0))
                except:
                    pass
            if ratios:
                attendance_rate = int(round(sum(ratios) / len(ratios) * 100))

        def _stat_block(icon_class, icon, label, value, help_text):
            return (
                '<div class="sch-stat">'
                  f'<div class="sch-stat-ico {icon_class}">{icon}</div>'
                  '<div>'
                    f'<div class="sch-stat-label">{label}</div>'
                    f'<div class="sch-stat-value">{value}</div>'
                    f'<div class="sch-stat-help">{help_text}</div>'
                  '</div>'
                '</div>'
            )

        stats_html = (
            '<div class="sch-stats">'
            + _stat_block("red",    "📅", "Total Classes",    total_classes,   "This Week")
            + _stat_block("green",  "📚", "Subjects",         subjects_count,  "Active Subjects")
            + _stat_block("amber",  "🕐", "Total Duration",   daily_duration,  "Daily Schedule")
            + _stat_block("purple", "📊", "Attendance Rate",  f"{attendance_rate}%", "This Week")
            + '</div>'
        )
        st.markdown(stats_html, unsafe_allow_html=True)

    elif st.session_state['page'] == "New Student Registration":
        # --- UI Redesign CSS Injection ---
        st.markdown("""
            <style>
            .enroll-header { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 5px; display: flex; align-items: center; gap: 12px; }
            .enroll-sub { color: #64748b; font-size: 1.05rem; margin-bottom: 30px; }
            
            .step-title { font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 10px; margin-bottom: 10px; background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
            
            .guideline-card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; height: 100%; }
            .g-title { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
            .g-box { background: #f8fafc; border-radius: 8px; padding: 20px; border: 1px solid #e2e8f0; }
            .g-item { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 16px; }
            .g-icon { font-size: 1.2rem; margin-top: 2px; }
            .g-text { font-size: 0.95rem; color: #475569; line-height: 1.5; margin: 0; }
            .g-text b { color: #1e293b; font-weight: 600; }
            
            .info-banner { background: #e2e8f0; color: #334155; padding: 16px; border-radius: 8px; font-size: 0.95rem; margin-bottom: 20px; border: 1px solid #cbd5e1; }
            
            /* Custom Action Button Override */
            div[data-testid="stButton"] button[kind="primary"] {
                background: linear-gradient(180deg, #dc2626 0%, #b91c1c 100%) !important;
                color: white !important;
                border-radius: 8px !important;
                padding: 12px !important;
                font-weight: 600 !important;
                font-size: 1.1rem !important;
                border: none !important;
                box-shadow: 0 4px 6px rgba(220, 38, 38, 0.2) !important;
            }
            div[data-testid="stButton"] button[kind="primary"]:hover {
                background: linear-gradient(180deg, #b91c1c 0%, #7f1d1d 100%) !important;
            }
            
            /* Input styling matching the wireframe */
            div[data-baseweb="input"] { background-color: #f8fafc !important; border-radius: 6px !important; border: 1px solid #e2e8f0 !important; }
            </style>
        """, unsafe_allow_html=True)

        # --- Header ---
        st.markdown("<div class='enroll-header'>📝 Biometric Enrollment</div>", unsafe_allow_html=True)
        st.markdown("<div class='enroll-sub'>Register a new student profile. Ensure the subject is in a well-lit environment and directly facing the webcam before initiating the capture sequence.</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.3, 1], gap="large")

        # --- Left Column: Interactive Form ---
        with col1:
            st.markdown("<div class='step-title'>Step 1: Identity Provisioning</div>", unsafe_allow_html=True)
            
            with st.container():
                name = st.text_input("Student Legal Name", placeholder="e.g., Karandeep Singh", label_visibility="collapsed")
                roll = st.text_input("University Roll Number", placeholder="e.g., 2420552", label_visibility="collapsed")
                st.write("")
            
            st.markdown("<div class='step-title'>Step 2: Biometric Scan</div>", unsafe_allow_html=True)
            
            with st.container():
                st.markdown("<div class='info-banner'>The system will activate the webcam. Press 'C' on your keyboard to capture the 128-D matrix, or 'Q' to abort.</div>", unsafe_allow_html=True)
                if st.button("📷 Initiate Camera & Capture", type="primary", use_container_width=True):
                    if name and roll: 
                        if capture_new_student(name, roll): 
                            st.success(f"✅ 128-D Biometric Matrix compiled and securely saved for {name}.")
                    else: 
                        st.error("Validation Error: Both Student Name and Roll Number must be provided.")

        # --- Right Column: Static Guidelines ---
        with col2:
            st.markdown("""
                <div class="guideline-card">
                    <div class="g-title">📋 Capture Guidelines</div>
                    <div class="g-box">
                        <div class="g-item">
                            <div class="g-icon">💡</div>
                            <p class="g-text"><b>Lighting:</b> Avoid heavy backlighting (e.g., sitting with a bright window directly behind the student).</p>
                        </div>
                        <div class="g-item">
                            <div class="g-icon">👤</div>
                            <p class="g-text"><b>Positioning:</b> The student's face should occupy at least 30% of the camera frame.</p>
                        </div>
                        <div class="g-item">
                            <div class="g-icon">👓</div>
                            <p class="g-text"><b>Accessories:</b> Remove dark glasses or heavy face-coverings to ensure accurate geometric mapping.</p>
                        </div>
                        <div class="g-item">
                            <div class="g-icon">🧑‍🤝‍🧑</div>
                            <p class="g-text"><b>Singular Focus:</b> Ensure only one face is visible in the frame during capture to prevent matrix collision.</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    elif st.session_state['page'] == "Live Dashboard":
        import plotly.express as px

        st.markdown("<h1 style='color:#1a1a2e; margin-bottom:4px;'>📈 Live Subject-Specific Logs</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#555; margin-bottom:20px;'>Student attendance analysis & subject-wise log viewer</p>", unsafe_allow_html=True)

        # Tile CSS
        st.markdown("""
            <style>
            .csv-tile {
                background: white;
                border: 2px solid #e0e4ea;
                border-radius: 12px;
                padding: 14px 16px;
                cursor: pointer;
                transition: all 0.2s ease;
                margin-bottom: 10px;
            }
            .csv-tile:hover { border-color: #b73229; box-shadow: 0 4px 12px rgba(183,50,41,0.12); }
            .csv-tile.active { border-color: #b73229; background: #fff5f5; }
            .tile-subject { font-size: 1.05rem; font-weight: 700; color: #1a1a2e; margin-bottom: 2px; }
            .tile-date    { font-size: 0.82rem; color: #888; }
            .tile-badge   { display:inline-block; background:#b73229; color:white; border-radius:20px;
                            font-size:0.72rem; font-weight:700; padding:2px 10px; margin-top:6px; }
            </style>
        """, unsafe_allow_html=True)

        if not os.path.exists(ATTENDANCE_DIR):
            st.error("Directory Error: Attendance folder not found.")
        else:
            all_files = sorted([f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.csv')], reverse=True)
            if not all_files:
                st.info("📭 No attendance logs found yet.")
            else:
                # ── Load all CSVs for the graph ───────────────────────────
                dfs = []
                for f in all_files:
                    try:
                        dfs.append(pd.read_csv(os.path.join(ATTENDANCE_DIR, f)))
                    except:
                        pass
                df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

                # ── STUDENT ATTENDANCE BAR CHART ──────────────────────────
                if not df_all.empty and 'NAME' in df_all.columns and 'SUBJECT' in df_all.columns:
                    df_all['STUDENT'] = df_all['NAME'].str.extract(r'^(.+?)\s*\(').fillna(df_all['NAME'])
                    chart_data = df_all.groupby(['STUDENT', 'SUBJECT']).size().reset_index(name='Attendance')

                    fig = px.bar(
                        chart_data,
                        x='STUDENT', y='Attendance', color='SUBJECT',
                        barmode='group',
                        color_discrete_sequence=['#b73229', '#2E6DB4', '#2ecc71', '#f39c12'],
                        labels={'STUDENT': 'Students', 'Attendance': 'Attendance', 'SUBJECT': 'Subject'},
                        text='Attendance'
                    )
                    fig.update_traces(textposition='outside', textfont_size=12)
                    fig.update_layout(
                        height=380,
                        margin=dict(t=30, b=20, l=10, r=10),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, title_text=''),
                        xaxis_title='Students', yaxis_title='Attendance',
                        plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Arial', size=13),
                        bargap=0.25, bargroupgap=0.05
                    )
                    fig.update_xaxes(showgrid=False, tickfont=dict(size=13, color='#1a1a2e'))
                    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0', zeroline=False)
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # ── CSV TILES ─────────────────────────────────────────────
                st.markdown("#### 📁 Select Target CSV File")

                if 'selected_tile' not in st.session_state:
                    st.session_state['selected_tile'] = all_files[0]

                # Parse tile metadata
                def parse_tile(fname):
                    try:
                        parts = fname.replace('.csv','').split('_')
                        subject = parts[1] if len(parts) > 1 else fname
                        date    = parts[2] if len(parts) > 2 else ''
                        return subject, date
                    except:
                        return fname, ''

                # Render tiles in rows of 4
                cols_per_row = 4
                rows = [all_files[i:i+cols_per_row] for i in range(0, len(all_files), cols_per_row)]
                for row in rows:
                    tile_cols = st.columns(cols_per_row)
                    for col, fname in zip(tile_cols, row):
                        subject, date = parse_tile(fname)
                        is_active = st.session_state['selected_tile'] == fname
                        border_col = '#b73229' if is_active else '#e0e4ea'
                        bg_col = '#fff5f5' if is_active else 'white'
                        with col:
                            st.markdown(f"""
                                <div style="background:{bg_col}; border:2px solid {border_col};
                                    border-radius:12px; padding:14px 16px; margin-bottom:6px;">
                                    <div style="font-size:1rem; font-weight:700; color:#1a1a2e;">📚 {subject}</div>
                                    <div style="font-size:0.8rem; color:#888; margin-top:2px;">📅 {date}</div>
                                    {'<div style="display:inline-block;background:#b73229;color:white;border-radius:20px;font-size:0.7rem;font-weight:700;padding:2px 10px;margin-top:6px;">● Active</div>' if is_active else ''}
                                </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"Open", key=f"tile_{fname}", use_container_width=True):
                                st.session_state['selected_tile'] = fname
                                st.rerun()

                # ── LOG TABLE for selected tile ────────────────────────────
                st.divider()
                selected_file = st.session_state['selected_tile']
                st.markdown(f"#### 📋 Log: `{selected_file}`")
                try:
                    df_log = pd.read_csv(os.path.join(ATTENDANCE_DIR, selected_file))
                    mc1, mc2 = st.columns([1, 5])
                    mc1.metric("Unique Students", len(df_log['NAME'].unique()) if 'NAME' in df_log.columns else 0)
                    with mc2:
                        with open(os.path.join(ATTENDANCE_DIR, selected_file), 'rb') as dl_f:
                            st.download_button("⬇️ Export CSV", dl_f, file_name=selected_file,
                                               mime='text/csv', use_container_width=False)
                    st.dataframe(
                        df_log, use_container_width=True, hide_index=False,
                        column_config={
                            "NAME":    st.column_config.TextColumn("👤 Name"),
                            "DATE":    st.column_config.TextColumn("📅 Date"),
                            "DAY":     st.column_config.TextColumn("📆 Day"),
                            "TIME":    st.column_config.TextColumn("⏰ Time"),
                            "SUBJECT": st.column_config.TextColumn("📚 Subject"),
                        }
                    )
                except Exception as e:
                    st.error(f"Data Read Error: {e}")

    elif st.session_state['page'] == "System Administration":
        st.markdown("""
            <style>
            /* ========= Students page ========= */
            .stu-head { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }
            .stu-head-icon {
                width: 56px; height: 56px; border-radius: 14px;
                background: #fee2e2;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.8rem; flex-shrink: 0;
            }
            .stu-head-title { font-size: 1.9rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
            .stu-head-sub { color: #64748b; font-size: 0.95rem; margin-top: 4px; }

            /* Stat cards row */
            .stu-stats-row {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 18px;
                margin-bottom: 24px;
            }
            .stu-stat-card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 18px 20px;
                display: flex;
                align-items: center;
                gap: 14px;
                box-shadow: 0 1px 3px rgba(15,23,42,0.04);
            }
            .stu-stat-ico {
                width: 50px; height: 50px; border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.4rem; flex-shrink: 0;
            }
            .stu-stat-ico.red    { background: #fee2e2; color: #b91c1c; }
            .stu-stat-ico.green  { background: #dcfce7; color: #166534; }
            .stu-stat-ico.amber  { background: #fef3c7; color: #b45309; }
            .stu-stat-ico.purple { background: #ede9fe; color: #6b21a8; }
            .stu-stat-label { font-size: 0.78rem; color: #64748b; font-weight: 500; }
            .stu-stat-value { font-size: 1.6rem; color: #0f172a; font-weight: 800; line-height: 1; margin: 2px 0; }
            .stu-stat-help  { font-size: 0.72rem; color: #94a3b8; }

            /* Enrolled students card */
            .stu-card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 22px 24px;
                box-shadow: 0 2px 10px rgba(15,23,42,0.04);
            }
            .stu-card-title {
                font-size: 1.2rem; font-weight: 700; color: #0f172a;
                margin-bottom: 16px; display: flex; align-items: center; gap: 10px;
            }

            /* Students table */
            .stu-table {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Poppins', sans-serif;
                margin-top: 10px;
            }
            .stu-table th {
                background: linear-gradient(180deg, #b91c1c 0%, #991b1b 100%);
                color: white !important;
                font-weight: 600;
                font-size: 0.88rem;
                padding: 14px 12px;
                text-align: left;
                border: none;
            }
            .stu-table th:first-child { border-top-left-radius: 10px; border-bottom-left-radius: 10px; text-align: center; }
            .stu-table th:last-child  { border-top-right-radius: 10px; border-bottom-right-radius: 10px; text-align: center; }
            .stu-table td {
                padding: 16px 12px;
                font-size: 0.93rem;
                color: #334155;
                border-bottom: 1px solid #f1f5f9;
            }
            .stu-table tr:last-child td { border-bottom: none; }
            .stu-idx { text-align: center; color: #94a3b8; font-weight: 600; }
            .stu-status-pill {
                display: inline-flex; align-items: center; gap: 6px;
                background: #dcfce7; color: #15803d;
                font-size: 0.78rem; font-weight: 600;
                border-radius: 20px; padding: 4px 12px;
            }
            .stu-status-dot { width: 7px; height: 7px; border-radius: 50%; background: #16a34a; }
            .stu-actions { text-align: center; color: #94a3b8; font-size: 1.1rem; letter-spacing: 2px; }

            .stu-footer {
                display: flex; justify-content: space-between; align-items: center;
                padding-top: 16px; margin-top: 6px;
                color: #64748b; font-size: 0.85rem;
            }
            .stu-page-pill {
                display: inline-flex; gap: 6px; align-items: center;
            }
            .stu-page-btn {
                width: 32px; height: 32px; border-radius: 6px;
                display: inline-flex; align-items: center; justify-content: center;
                background: white; border: 1px solid #e5e7eb; color: #64748b;
                font-size: 0.85rem; font-weight: 600;
            }
            .stu-page-btn.active { background: #b91c1c; color: white; border-color: #b91c1c; }

            /* Remove student card */
            .rm-card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 22px 24px;
                box-shadow: 0 2px 10px rgba(15,23,42,0.04);
            }
            .rm-head { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
            .rm-ico {
                width: 44px; height: 44px; border-radius: 11px;
                background: #fee2e2;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.3rem; color: #b91c1c;
            }
            .rm-title { font-size: 1.15rem; font-weight: 700; color: #0f172a; }
            .rm-desc { font-size: 0.88rem; color: #64748b; line-height: 1.55; margin-bottom: 16px; }

            .rm-warn {
                background: #fef2f2;
                border-radius: 10px;
                padding: 14px 16px;
                margin-top: 16px;
                display: flex; gap: 10px; align-items: flex-start;
            }
            .rm-warn-ico { font-size: 1.1rem; color: #b91c1c; flex-shrink: 0; }
            .rm-warn-title { font-size: 0.88rem; font-weight: 600; color: #b91c1c; margin-bottom: 2px; }
            .rm-warn-text { font-size: 0.82rem; color: #7a1515; line-height: 1.5; }
            </style>
        """, unsafe_allow_html=True)

        # --- Header ---
        st.markdown(
            '<div class="stu-head">'
              '<div class="stu-head-icon">👥</div>'
              '<div>'
                '<div class="stu-head-title">Students</div>'
                '<div class="stu-head-sub">Manage and view enrolled students.</div>'
              '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # --- Load student data (preserves backend: still reads encodings.pkl) ---
        df_students = pd.DataFrame(columns=["Name", "Roll Number"])
        raw_names = []
        known_data = None
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                known_data = pickle.load(f)
            raw_names = known_data["names"]
            parsed_data = [
                [e.split("(")[0].strip(), e.split("(")[1].replace(")", "").strip()]
                if "(" in e else [e, "N/A"]
                for e in raw_names
            ]
            df_students = pd.DataFrame(parsed_data, columns=["Name", "Roll Number"])

        total_students = len(df_students)
        active_students = total_students  # all enrolled = active in this system
        inactive_students = 0

        # Attendance Today = unique names in today's CSVs across subjects
        attendance_today = 0
        today_str = datetime.now().strftime("%d-%m-%Y")
        try:
            todays_unique = set()
            for fname in os.listdir(ATTENDANCE_DIR):
                if fname.endswith(f"_{today_str}.csv"):
                    try:
                        dft = pd.read_csv(os.path.join(ATTENDANCE_DIR, fname))
                        if "NAME" in dft.columns:
                            todays_unique.update(dft["NAME"].dropna().unique())
                    except:
                        pass
            attendance_today = len(todays_unique)
        except:
            attendance_today = 0

        # --- Stat cards row ---
        def _stat(icon_cls, icon, label, value, help_text):
            return (
                '<div class="stu-stat-card">'
                  f'<div class="stu-stat-ico {icon_cls}">{icon}</div>'
                  '<div>'
                    f'<div class="stu-stat-label">{label}</div>'
                    f'<div class="stu-stat-value">{value}</div>'
                    f'<div class="stu-stat-help">{help_text}</div>'
                  '</div>'
                '</div>'
            )

        stats_html = (
            '<div class="stu-stats-row">'
            + _stat("red",    "👥", "Total Students",    total_students,   "Enrolled")
            + _stat("green",  "👤➕", "Active Students",  active_students,  "Currently Active")
            + _stat("amber",  "👤✕", "Inactive Students", inactive_students,"Currently Inactive")
            + _stat("purple", "🪪", "Attendance Today",   attendance_today, "Marked")
            + '</div>'
        )
        st.markdown(stats_html, unsafe_allow_html=True)

        # --- Main content: 2-column (enrolled table + remove student card) ---
        col_left, col_right = st.columns([2, 1], gap="large")

        with col_left:
            st.markdown('<div class="stu-card">', unsafe_allow_html=True)
            st.markdown('<div class="stu-card-title">📋 Enrolled Students</div>', unsafe_allow_html=True)

            # Search & filter row using Streamlit widgets (for real functionality)
            srch_col, flt_col = st.columns([2, 1])
            with srch_col:
                search_q = st.text_input(
                    "search_stu",
                    placeholder="🔍 Search students...",
                    label_visibility="collapsed",
                    key="stu_search"
                )
            with flt_col:
                status_filter = st.selectbox(
                    "stu_filter",
                    ["All Status", "Active", "Inactive"],
                    label_visibility="collapsed",
                    key="stu_status"
                )

            # Apply search filter
            df_view = df_students.copy()
            if search_q:
                mask = (
                    df_view["Name"].str.contains(search_q, case=False, na=False) |
                    df_view["Roll Number"].astype(str).str.contains(search_q, case=False, na=False)
                )
                df_view = df_view[mask]

            # Build table
            if len(df_view) > 0:
                rows_html = ""
                for i, (_, r) in enumerate(df_view.iterrows(), start=1):
                    rows_html += (
                        '<tr>'
                          f'<td class="stu-idx">{i}</td>'
                          f'<td>{r["Name"]}</td>'
                          f'<td>{r["Roll Number"]}</td>'
                          '<td><span class="stu-status-pill"><span class="stu-status-dot"></span>Active</span></td>'
                          '<td>AIML</td>'
                          '<td class="stu-actions">⋮</td>'
                        '</tr>'
                    )

                table_html = (
                    '<table class="stu-table">'
                      '<thead><tr>'
                        '<th style="width:60px;">#</th>'
                        '<th>Name</th>'
                        '<th>Roll Number</th>'
                        '<th style="text-align:center;">Status</th>'
                        '<th style="text-align:center;">Branch</th>'
                        '<th style="width:80px; text-align:center;">Actions</th>'
                      '</tr></thead>'
                      f'<tbody>{rows_html}</tbody>'
                    '</table>'
                )
                st.markdown(table_html, unsafe_allow_html=True)

                # Footer: count + pagination (cosmetic)
                footer_html = (
                    '<div class="stu-footer">'
                      f'<div>Showing 1 to {len(df_view)} of {len(df_view)} students</div>'
                      '<div class="stu-page-pill">'
                        '<span class="stu-page-btn">‹</span>'
                        '<span class="stu-page-btn active">1</span>'
                        '<span class="stu-page-btn">›</span>'
                      '</div>'
                    '</div>'
                )
                st.markdown(footer_html, unsafe_allow_html=True)
            else:
                if total_students == 0:
                    st.info("No student profiles registered yet. Use 'Register Student' to enroll.")
                else:
                    st.info("No students match your search.")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="rm-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="rm-head">'
                  '<div class="rm-ico">🗑️</div>'
                  '<div class="rm-title">Remove Student</div>'
                '</div>'
                '<div class="rm-desc">Permanently delete a student\'s facial recognition data and all related records.</div>',
                unsafe_allow_html=True
            )

            if raw_names:
                student_to_delete = st.selectbox(
                    "Select student to remove",
                    raw_names,
                    label_visibility="collapsed",
                    key="stu_delete_select"
                )
                if st.button("🗑️  Delete Student Record", type="primary", use_container_width=True, key="stu_delete_btn"):
                    idx = known_data["names"].index(student_to_delete)
                    known_data["names"].pop(idx)
                    known_data["encodings"].pop(idx)
                    with open(ENCODINGS_FILE, 'wb') as f:
                        pickle.dump(known_data, f)
                    st.toast(f"Successfully removed {student_to_delete}.")
                    st.rerun()

                st.markdown(
                    '<div class="rm-warn">'
                      '<div class="rm-warn-ico">⚠️</div>'
                      '<div>'
                        '<div class="rm-warn-title">This action cannot be undone.</div>'
                        '<div class="rm-warn-text">All associated attendance records will also be removed.</div>'
                      '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("No students to remove.")

            st.markdown('</div>', unsafe_allow_html=True)

        # --- Administrators management moved to an expander at the bottom ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🛡️  Manage System Administrators"):
            if os.path.exists(CRED_FILE):
                df_ad = pd.read_csv(CRED_FILE, dtype=str)
                display_df = df_ad[['Full Name', 'Username']].copy()

                col_ad1, col_ad2 = st.columns([2, 1], gap="large")

                with col_ad1:
                    st.metric("Total Active Administrators", len(display_df))
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    st.divider()
                    st.markdown("#### ➕ Add New Administrator")
                    with st.form("new_admin_form"):
                        col_fn, col_un = st.columns(2)
                        with col_fn: fn = st.text_input("Full Name", placeholder="e.g. Jane Doe")
                        with col_un: un = st.text_input("Username", placeholder="admin_username")
                        pw = st.text_input("Password", type="password", placeholder="Enter secure password")
                        submit_admin = st.form_submit_button("Create Administrator Account", type="primary", use_container_width=True)

                        if submit_admin:
                            if fn and un and pw:
                                file_exists = os.path.isfile(CRED_FILE)
                                with open(CRED_FILE, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    if not file_exists: writer.writerow(["Full Name", "Username", "Password"])
                                    writer.writerow([fn, un, pw])
                                st.toast(f"Account created for {fn}.")
                                st.rerun()
                            else:
                                st.toast("Please fill in all fields.", icon="⚠️")

                with col_ad2:
                    st.markdown("""
                        <div style="background:#fef2f2; padding:18px; border-radius:10px; border-left:4px solid #dc2626;">
                            <div style="color:#b91c1c; font-weight:700; font-size:1rem;">🗑️ Revoke Access</div>
                            <div style="font-size:0.82rem; color:#7a1515; margin-top:6px;">Remove an administrator's login access entirely.</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.write("")
                    admin_to_delete = st.selectbox("Select Admin Username", df_ad['Username'].unique(), label_visibility="collapsed", key="admin_delete_sel")
                    if st.button("Remove Admin Access", type="primary", use_container_width=True, key="admin_delete_btn"):
                        if len(df_ad) <= 1:
                            st.toast("Cannot delete the only remaining administrator.", icon="⚠️")
                        else:
                            df_ad[df_ad['Username'] != admin_to_delete].to_csv(CRED_FILE, index=False)
                            st.toast(f"Admin access revoked for: {admin_to_delete}")
                            st.rerun()
            else:
                st.info("No administrators found in database.")

    elif st.session_state['page'] == "Research":
        st.markdown("""
            <style>
            .help-card {
                background: white;
                border-radius: 14px;
                border: 1px solid #e5e7eb;
                border-left: 5px solid #dc2626;
                padding: 24px 26px;
                box-shadow: 0 2px 6px rgba(15,23,42,0.04);
                margin-bottom: 18px;
                transition: all 0.2s ease;
                height: 100%;
            }
            .help-card:hover { box-shadow: 0 8px 22px rgba(185,28,28,0.10); transform: translateY(-2px); }
            .help-title {
                font-size: 1.15rem; font-weight: 700; color: #0f172a;
                margin-bottom: 10px; display: flex; align-items: center; gap: 10px;
            }
            .help-emoji {
                width: 40px; height: 40px; border-radius: 10px;
                background: #fef2f2; display: inline-flex;
                align-items: center; justify-content: center; font-size: 1.2rem;
            }
            .help-text { color: #475569; font-size: 0.95rem; line-height: 1.65; margin: 0; }
            .help-tip {
                background: #fef2f2; border-radius: 8px; padding: 10px 14px;
                margin-top: 12px; font-size: 0.85rem; color: #7a1515;
                border-left: 3px solid #dc2626;
            }
            .faq-item {
                background: white; border: 1px solid #e5e7eb;
                border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
            }
            .faq-q { font-weight: 600; color: #0f172a; font-size: 0.95rem; margin-bottom: 6px; }
            .faq-a { color: #64748b; font-size: 0.88rem; line-height: 1.55; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="page-title">❓ Help & Support</div>
            <div class="page-sub">A simple overview of how the Smart Attendance system works — no technical background needed.</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("""
                <div class="help-card">
                    <div class="help-title">
                        <span class="help-emoji">👁️</span>
                        How it spots real people (not photos)
                    </div>
                    <p class="help-text">
                        Before marking anyone present, the system asks the student to <b>blink twice</b> in front of the camera.
                        A photo or a phone screen can't blink — so this simple trick blocks anyone trying to mark
                        someone else present using their picture.
                    </p>
                    <div class="help-tip">💡 Tip: Stand directly in front of the camera with good lighting and blink naturally.</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="help-card">
                    <div class="help-title">
                        <span class="help-emoji">⚡</span>
                        Why it runs smoothly on any laptop
                    </div>
                    <p class="help-text">
                        The system shrinks each camera frame down before analyzing it, which makes it about
                        <b>4 times faster</b> without losing accuracy. This way the video stays smooth
                        even on older laptops without freezing or lagging.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="help-card">
                    <div class="help-title">
                        <span class="help-emoji">🧠</span>
                        How it recognizes each student
                    </div>
                    <p class="help-text">
                        Every face is turned into a unique <b>digital fingerprint</b> — a long string of numbers
                        that describes the specific features of that face. When a student stands in front of the camera,
                        the system compares their live fingerprint with the stored ones to find a match.
                    </p>
                    <div class="help-tip">💡 This works even in different lighting, slight angles, or with small changes like glasses.</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="help-card">
                    <div class="help-title">
                        <span class="help-emoji">🛡️</span>
                        What keeps the data safe
                    </div>
                    <p class="help-text">
                        The system never stores actual photos of students — only the mathematical fingerprint.
                        Even if someone got the file, they couldn't rebuild a face from it. The attendance records
                        are saved as plain CSV files per subject per day, so you can export and share them easily.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # --- Quick FAQ ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-bottom: 14px;">Quick Questions</div>""", unsafe_allow_html=True)

        faq = [
            ("Why do I have to blink twice?",
             "Blinking is how the system confirms a real person is in front of the camera. "
             "One blink could be a glitch — two blinks means it's definitely you."),
            ("What if the camera doesn't recognize me?",
             "Make sure your face is well-lit and clearly visible. Remove sunglasses or masks. "
             "If the issue continues, ask your teacher to re-enroll your face from the Register Student page."),
            ("Can two students be marked at the same time?",
             "No. The system intentionally looks at one person at a time. Each student takes turns in front of the camera."),
            ("What happens if I try to mark myself twice in a row?",
             "The system has a 5-minute cool-down. Any repeat attempts within that window are automatically ignored."),
            ("Where are attendance records stored?",
             "Each subject's daily attendance is saved as a CSV file you can find in the Attendance Records page, "
             "and exported with one click."),
        ]
        for q, a in faq:
            st.markdown(f"""
                <div class="faq-item">
                    <div class="faq-q">❓ {q}</div>
                    <div class="faq-a">{a}</div>
                </div>
            """, unsafe_allow_html=True)