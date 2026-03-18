import streamlit as st
import pandas as pd
import os

# Page styling
st.set_page_config(page_title="Login - Face Recognition System")
st.markdown("<h1 style='text-align: center;'>🔐 User Login</h1>", unsafe_allow_html=True)

# Define the path to your credentials
CRED_FILE = "data/user_credentials.csv"

# UI for the Login Form
with st.container():
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Check if the database exists
        if not os.path.exists(CRED_FILE):
            st.error("No registered users found. Please register first!")
        else:
            # Load the credentials
            users_df = pd.read_csv(CRED_FILE, dtype=str)
            
            # Search for the user in the CSV
            user_record = users_df[(users_df['Username'] == username_input) & 
                                   (users_df['Password'] == str(password_input))]
            
            if not user_record.empty:
                st.success(f"Welcome back, {user_record.iloc[0]['Full Name']}!")
                st.balloons()
                st.info("You are now authorized to use the Attendance System.")
                # In a real app, you would redirect to app.py here
            else:
                st.error("Invalid Username or Password. Please try again.")

# Optional link to Registration
if st.button("Don't have an account? Register here"):
    st.info("Run 'streamlit run register.py' to create a new account.")