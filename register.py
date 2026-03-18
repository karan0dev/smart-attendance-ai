import streamlit as st
import cv2
import pickle
import numpy as np
import os
import csv
import time

# Page Configuration
st.set_page_config(page_title="Face Recognition System", layout="centered")

# Custom CSS for the "Registration Card" look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #00bcd4; color: white; }
    .register-btn>button { background-color: #3f51b5 !important; }
    </style>
""", unsafe_allow_html=True)

# UI Header
st.markdown("<h1 style='text-align: center;'>🎓 User Registration</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Create your account to access the system</p>", unsafe_allow_html=True)

# Registration Form Card
with st.container():
    full_name = st.text_input("Full Name", placeholder="Enter your full name")
    username = st.text_input("Username", placeholder="Choose a username")
    password = st.text_input("Password", type="password", placeholder="Create a password")
    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

    # ... (User input fields for name, username, etc.) ...

# ... (Full Name, Username, etc. inputs are above)

# ... (Keep your existing Full Name, Username, and Password inputs above) ...

# --- Line 65: Final Registration Button ---
if st.button("Register Account"):
    # Safety Check: Ensure no fields are empty
    if not full_name or not username or not password:
        st.error("Please fill in all fields before registering.")
    elif password != confirm_password:
        st.error("Passwords do not match! Please check again.")
    else:
        # Step 1: Ensure the 'data' folder exists
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Step 2: Define the credential file path
        cred_file = "data/user_credentials.csv"
        file_exists = os.path.isfile(cred_file)

        # Step 3: Write the data to the CSV
        try:
            with open(cred_file, "a", newline="") as f:
                writer = csv.writer(f)
                # If the file is new, add the header row first
                if not file_exists:
                    writer.writerow(["Full Name", "Username", "Password"])
                writer.writerow([full_name, username, password])
            
            st.balloons() # Visual celebration!
            st.success(f"Account for {username} created successfully!")
            st.info("You can now go to the Login page.")
        except Exception as e:
            st.error(f"Error saving credentials: {e}")