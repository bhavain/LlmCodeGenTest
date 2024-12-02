import streamlit as st
import subprocess
import pandas as pd
import zipfile
import plotly.express as px

# Define paths and configurations
METADATA_FOLDER = "metadata"
SUBMISSIONS_FOLDER = "problem_submissions"
BENCHMARK_FILE = "benchmark_results.csv"

# Load problem list
@st.cache_data
def load_problem_list():
    # Assuming problem_list.csv contains a mapping of problem_id and name
    return pd.read_csv("problem_list.csv")

def process_bulk_problems():
    # Simulate loading the problem list from a file or database
    problem_list = ["p00001", "p00002", "p00003", "p00004", "p00005", "p00006", "p00007", "p00008", "p00009", "p00010", 
                    "p00011", "p00012", "p00015", "p00017", "p00018", "p00019", "p00028", "p00029", "p03992", "p03993", 
                    "p03997", "p03999"]
    # problem_list = ["p00001"]
    return problem_list

# Load problem description HTML
def load_problem_description(problem_id):
    problem_desc_path = f"problem_descriptions/{problem_id}.html"
    with zipfile.ZipFile("problem_descriptions.zip", "r") as zip_ref:
        with zip_ref.open(problem_desc_path, 'r') as file:
            content = file.read()
    return content.decode('utf-8')

def run_automation():
    problems = process_bulk_problems()  # Load list of problems
    total_problems = len(problems)
    
    progress = st.progress(0)  # Progress bar

    for i, problem_id in enumerate(problems):
        if st.session_state.stop_automation:
            st.error("Automation stopped by user!")
            break  # Exit the loop if stop flag is set

        # st.write(f"Processing problem {problem_id}...")
        # Simulate processing (replace with actual automation logic)
        subprocess.run(f"python3 ./scripts/test.py --problem_id {problem_id}", shell=True)

        # Update progress
        progress.progress((i + 1) / total_problems, text=f"Processing {problem_id} ({i + 1}/{total_problems})")
    
    if not st.session_state.stop_automation:
        st.success("Automation completed!")

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

    # Initialize session state
    if "stop_automation" not in st.session_state:
        st.session_state.stop_automation = False

    if st.button("Start Automation"):
        st.session_state.stop_automation = False
        run_automation()

    if st.button("Stop Automation"):
        st.session_state.stop_automation = True
        st.warning("Automation stopped!")

    if st.button("Show Metrics Summary"):
        df = pd.read_csv(BENCHMARK_FILE)
        st.subheader("Benchmark Metrics Summary")
        
        total_problems = len(df)
        avg_complexity = df['average_case_complexity'].mean()
        avg_pass_rate = (df['average_passed_cases'] / df['total_cases']).mean() * 100
        avg_fail_rate = (df['average_failed_cases'] / df['total_cases']).mean() * 100
        
        st.metric("Total Problems Processed", total_problems)
        st.metric("Average Case Complexity", f"{avg_complexity:.2f}")
        st.metric("Average Pass Rate", f"{avg_pass_rate:.2f}%")
        st.metric("Average Fail Rate", f"{avg_fail_rate:.2f}%")

    if st.button("Visualize Problem Performance"):
        df = pd.read_csv("benchmark_results.csv")
        st.subheader("Problem-Level Performance")
        
        # Bar chart: Passed vs Failed Cases
        st.bar_chart(df[['average_passed_cases', 'average_failed_cases']])
        
        # Scatter plot: Complexity vs Lines of Code
        fig = px.scatter(
            df,
            x="average_case_complexity",
            y="average_lines_of_code",
            size="total_cases",
            color="problem_id",
            title="Complexity vs Lines of Code"
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
