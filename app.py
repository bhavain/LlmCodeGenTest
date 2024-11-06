import streamlit as st
import subprocess
import pandas as pd
import zipfile
import random

# Define paths and configurations
METADATA_FOLDER = "metadata"
SUBMISSIONS_FOLDER = "problem_submissions"
LOG_FILE = "benchmark_results.csv"

# Load problem list
@st.cache_data
def load_problem_list():
    # Assuming problem_list.csv contains a mapping of problem_id and name
    return pd.read_csv("problem_list.csv")

def process_bulk_problems():
    # Simulate loading the problem list from a file or database
    # problem_list = ["p00001", "p00002", "p00003", "p00004", "p00005", "p00006", "p00007", "p00008", "p00009", "p00010",
    #                 "p00011", "p00012", "p00013", "p00014", "p00015", "p00016", "p00017", "p00018", "p00019", "p00020"]
    
    problem_list = ["p00001", "p00002", "p00003", "p00004", "p00005", "p00006", "p00007", "p00008", "p00009", "p00010", "p00011", "p00012", "p00015", "p00017", "p00018", "p00019", "p00028"]
    return problem_list

# Process 20 problems and generate 25 test cases for each problem
def start_automation(problem_id):
    # Run the automation script for the selected problem

    selected_problems = process_bulk_problems()
    # selected_problems = random.sample(problem_list, num_problems)

    for problem_id in selected_problems:

        # Generate and save test cases for the problem
        subprocess.run(f"python3 ./scripts/test.py --problem_id {problem_id}", shell=True)

# Load problem description HTML
def load_problem_description(problem_id):
    problem_desc_path = f"problem_descriptions/{problem_id}.html"
    with zipfile.ZipFile("problem_descriptions.zip", "r") as zip_ref:
        with zip_ref.open(problem_desc_path, 'r') as file:
            content = file.read()
    return content.decode('utf-8')

# Streamlit app layout
def main():
    st.title("LLM Code Generation & Benchmark Dashboard")

    # Load the problem list
    problem_list = load_problem_list()

    # Display dropdown to select problem
    selected_problem_id = st.selectbox("Select Problem", problem_list['id'])

    # Display problem description
    st.subheader("Problem Description")
    problem_description = load_problem_description(selected_problem_id)
    st.markdown(problem_description, unsafe_allow_html=True)

    # Start button to trigger automation process
    if st.button("Start Automation"):
        st.write(f"Starting automation for problem {selected_problem_id}...")
        start_automation(selected_problem_id)
        st.success("Automation process completed!")

    # Display benchmark results
    if st.button("Show Benchmark Results"):
        df = pd.read_csv(LOG_FILE)
        st.write(df)
        # st.bar_chart(df[['passed_cases', 'failed_cases']])

if __name__ == "__main__":
    main()
