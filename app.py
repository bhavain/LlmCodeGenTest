import streamlit as st
import pandas as pd
from scripts.compile_and_test import filter_submissions, test_submissions, generate_unit_tests

# Load problem list CSV
@st.cache_data
def load_problem_list(file):
    return pd.read_csv(file)

# Load problem description HTML
def load_problem_description(problem_id):
    problem_desc_path = f"problem_descriptions/{problem_id}.html"
    with open(problem_desc_path, 'r') as file:
        return file.read()

# Load metadata for selected problem
@st.cache_data
def load_problem_metadata(problem_id):
    metadata_path = f"metadata/{problem_id}.csv"
    return pd.read_csv(metadata_path)

# Main app function
def main():
    st.title("Submission Testing Dashboard")
    
    # Load problem list
    problem_list_file = "problem_list.csv"
    problem_list = load_problem_list(problem_list_file)
    
    # Display problem selection
    selected_problem_id = st.selectbox("Select Problem", problem_list['id'])
    
    # Display problem description
    st.subheader("Problem Description")
    problem_description = load_problem_description(selected_problem_id)
    st.markdown(problem_description, unsafe_allow_html=True)
    
    # Load and display problem metadata
    # problem_metadata = load_problem_metadata(selected_problem_id)
    # st.subheader("Problem Submission Metadata")
    # st.write(problem_metadata.head(10))  # Display top 10 entries

    
    # Button to generate unit tests and run tests
    if st.button("Generate Unit Test Cases and Test Submissions"):
        # Generate unit test cases using Ollama
        unit_tests = generate_unit_tests(problem_description)
        st.subheader("Generated Unit Test Cases")
        st.write(unit_tests)
        # st.code(unit_tests)  # Display generated test cases

        # Filter and test submissions
        # submission_metadata = filter_submissions(f"metadata/{selected_problem_id}.csv")
        # test_submissions(selected_problem_id, submission_metadata, unit_tests)
        # st.success(f"Tests completed for problem {selected_problem_id}!") 
    
if __name__ == "__main__":
    main()
