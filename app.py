import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import os

# Get current date and time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

# Auto-refresh setup
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# Display count with FizzBuzz logic
if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Define the path for the attendance file
csv_file_path = f"Attendance/Attendance_{date}.csv"

# Check if the attendance file exists
if os.path.isfile(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    # Display the dataframe with highlighted max values
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.write(f"No attendance data available for {date}.")
