import os
import subprocess
import pandas as pd
import ollama

def filter_submissions(problem_metadata_file, language='C++', status='Accepted'):
    # Load problem-specific metadata CSV
    df = pd.read_csv(problem_metadata_file)
    
    # Filter C++ and Accepted submissions
    filtered_df = df[(df['language'] == language) & (df['status'] == status)]
    
    # Select 5 random submissions
    random_submissions = filtered_df.sample(n=min(5, len(filtered_df)))
    return random_submissions

def compile_submission(submission_file):
    output_file = submission_file.replace(".cpp", ".out")
    compile_cmd = f"g++ {submission_file} -o {output_file}"
    
    try:
        subprocess.run(compile_cmd, check=True, shell=True)
        print(f"Compilation successful: {submission_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {submission_file}")
        return None

def run_test(compiled_file, input_file, expected_output_file):
    with open(input_file, 'r') as input_data:
        result = subprocess.run([compiled_file], input=input_data.read(), capture_output=True, text=True)
        output = result.stdout
    
    with open(expected_output_file, 'r') as expected_output:
        expected = expected_output.read().strip()
    
    return output.strip() == expected

def run_test(compiled_file, unit_tests):
    """
    Simulate running the generated unit tests.
    Since there is no input/output data, we'll assume unit_tests contain function calls.
    """
    for test in unit_tests:
        print(f"Running test: {test}")
        result = subprocess.run([compiled_file], input=test, capture_output=True, text=True)
        print(f"Test Output: {result.stdout}")

def test_submissions(problem_id, submission_metadata, unit_tests):
    problem_folder = f"problem_submissions/{problem_id}/"
    
    for idx, row in submission_metadata.iterrows():
        submission_path = os.path.join(problem_folder, f"{row['submission_id']}.cpp")
        compiled_file = compile_submission(submission_path)
        
        if compiled_file:
            # Call Ollama to generate unit test cases and run tests here
            # is_correct = run_test(compiled_file)
            is_correct = run_test(compiled_file, unit_tests)
            # print(f"Test result for {submission_path}: {'Passed' if is_correct else 'Failed'}")
            print(f"Running tests for: {submission_path}")
        else:
            print(f"Skipping testing for {submission_path}")

def generate_unit_tests(problem_description):
    """
    Use Ollama to generate unit test cases based on the problem description.
    """
    response = ollama.chat(model="llama3.1", messages=[
        {
            "role": "user", 
            "content": f"""
            You are a code generation expert. Generate 5 unit test cases for the following problem:
            {problem_description}
            """
        }
    ])
    return response['message']['content']