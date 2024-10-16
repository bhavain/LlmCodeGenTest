import os
import subprocess
import pandas as pd
import ollama
import argparse
import time
import re

# Logging function to track benchmarking results
def log_benchmark(problem_id, total_submissions, valid_submissions, total_cases, average_passed_cases, average_failed_cases):
    log_file = "benchmark_results.csv"
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("problem_id,total_submissions,valid_submissions,total_cases,average_passed_cases,average_failed_cases,time_taken\n")
    with open(log_file, 'a') as f:
        f.write(f"{problem_id},{total_submissions},{valid_submissions},{total_cases},{average_passed_cases},{average_failed_cases}\n")

# Compile and test submission
def compile_submission(submission_file):
    output_file = submission_file.replace(".cpp", ".out")
    compile_cmd = f"g++ {submission_file} -o {output_file}"
    
    try:
        subprocess.run(compile_cmd, check=True, shell=True)
        return output_file
    except subprocess.CalledProcessError:
        return None

def run_test(compiled_file, input_file, expected_output_file):
    """
    Runs the compiled program using the input from input_file and compares
    the output with expected_output_file. Includes a timeout of 10 seconds.
    If execution exceeds 10 seconds, the test is marked as failed and skipped.
    """
    try:
        timeout = 10
        with open(input_file, 'r') as input_data:
            # Run the compiled file with the input and capture the output, adding a timeout of 10 seconds
            result = subprocess.run([compiled_file], input=input_data.read(), capture_output=True, text=True, timeout=timeout, check=True)
            output = result.stdout.strip()

        with open(expected_output_file, 'r') as expected_output:
            expected = expected_output.read().strip()

        return output == expected

    except subprocess.TimeoutExpired:
        # TimeoutExpired will be raised if the test exceeds the timeout
        print(f"Test for {compiled_file} timed out after {timeout} seconds. Skipping this test.")
        return False
    
    except subprocess.CalledProcessError:
        # CalledProcessError will be raised if the program returns a non-zero exit code
        return False

# Load metadata
def load_problem_metadata(problem_id, language = 'C++', status = "Accepted"):
    metadata_file = f"metadata/{problem_id}.csv"
    df = pd.read_csv(metadata_file)
    filtered_df = df[(df['language'] == language) & (df['status'] == status)]
    return filtered_df

def generate_and_save_test_cases(problem_id, problem_description):
    """
    Use Ollama to generate input and output test cases based on the problem description,
    and save them as files under the problem_tests folder for the given problem_id.
    """
    
    # Define the directory for test cases
    problem_test_dir = f"problem_tests_generated/{problem_id}"
    os.makedirs(problem_test_dir, exist_ok=True)
    
    # Check how many test cases already exist
    # existing_inputs = [f for f in os.listdir(problem_test_dir) if f.startswith('input_')]
    # existing_test_cases_count = len(existing_inputs)
    
    # # If we already have 5 test cases, do nothing
    # if existing_test_cases_count >= 5:
    #     print(f"Test case limit reached for problem {problem_id}. No new test cases generated.")
    #     return
    
    # # Otherwise, generate only the remaining test cases needed
    remaining_test_cases = 20 #5 - existing_test_cases_count
    # print(f"Generating {remaining_test_cases} new test cases for problem {problem_id}...")

    # Prompt Ollama to generate the remaining test cases
    response = ollama.chat(model="llama3.1", messages=[
        {
            "role": "user", 
            "content": f"""
            You are a code generation expert. Generate {remaining_test_cases} test cases with structured input and expected output
            for the following problem:
            
            {problem_description}
            
            Each test case should only have:
            - An "Input" section containing the input values.
            - An "Expected Output" section with the output value.

            The format of each test case should be:
        
            Test Case X:
            Input: <formatted input here>
            Expected Output: <formatted output here>
            
            Please ensure that the input is simple and easy to parse, and the output should match the format.
            Do not include any descriptions, explanations, or additional text.
            """
        }
    ])
    
    # Extract input/output from response
    test_cases = response['message']['content']
    inputs, outputs = extract_input_output(test_cases)  # Implement this to parse the test cases

    # Save the new test cases
    for i, (input_data, output_data) in enumerate(zip(inputs, outputs), start=1):
        input_file = os.path.join(problem_test_dir, f"input_{i}.txt")
        output_file = os.path.join(problem_test_dir, f"output_{i}.txt")
        
        # Save input
        with open(input_file, 'w') as f:
            f.write(input_data)
        
        # Save output
        with open(output_file, 'w') as f:
            f.write(output_data)
    
    print(f"{remaining_test_cases} new test cases saved for problem {problem_id} in {problem_test_dir}")
    # return inputs, outputs

def extract_input_output(test_cases):
    """
    Parse the test cases from the Ollama response to extract inputs and outputs.
    Test cases follow the pattern:
    - Input <input value>
    - Expected Output <output value>
    """

    inputs = []
    outputs = []

    # Regex to extract input and output from the test cases
    # test_case_pattern = re.compile(r"Input\s*```(?:markdown)?\s*(.*?)\s*```.*?Expected Output\s*```(?:markdown)?\s*(.*?)\s*```", re.DOTALL)
    # Regex to capture the input and output sections between backticks or quotes
    # test_case_pattern = re.compile(r"Input: [`'](.*?)['`].*?Expected Output: [`'](.*?)['`]", re.DOTALL)
    test_case_pattern = re.compile(
        r"(?:###|Input:?|input:?|`Input`|Input\s*```|Input\s*\*\*)\s*[\(\[\{]*\s*(.*?)\s*[\)\]\}]*\s*(?:###|Expected Output:?|expected output:?|`Expected Output`|Output\s*```|Output\s*\*\*)\s*[\(\[\{]*\s*(.*?)\s*[\)\]\}]*\s*(?=\*\*Test Case|\Z)", 
        re.DOTALL | re.IGNORECASE
    )


    # Find all matches for input/output pairs
    matches = test_case_pattern.findall(test_cases)
    
    for match in matches:
        input_part = match[0].replace('**', '').replace('`', '').strip()  # Clean the input value
        output_part = match[1].replace('**', '').replace('`', '').strip()  # Clean the output value
        
        inputs.append(input_part)
        outputs.append(output_part)

    return inputs, outputs

def filter_submissions(problem_id, language='C++', status='Accepted'):
    # Load problem-specific metadata CSV
    df = load_problem_metadata(problem_id)
    
    # Filter C++ and Accepted submissions
    filtered_df = df[(df['language'] == language) & (df['status'] == status)]
    
    # Select 5 random submissions
    random_submissions = filtered_df.sample(n=min(5, len(filtered_df)))
    return random_submissions

def test_submissions(problem_id, submission_metadata):
    problem_folder = f"problem_submissions_part_2/{problem_id}/"
    test_cases_folder = f"problem_tests_generated/{problem_id}/"

    total_submissions = len(submission_metadata)
    valid_submissions = 0
    total_passed_cases = 0
    total_failed_cases = 0
    max_submissions = 20  # Limit to 20 submissions

    test_case_files = [f for f in os.listdir(test_cases_folder) if 'input_' in f]
    total_cases = len(test_case_files)

    for idx, row in submission_metadata.iterrows():
        submission_path = os.path.join(problem_folder, f"{row['submission_id']}.cpp")
        if not os.path.exists(submission_path):
            print(f"Submission file not found: {submission_path}")
            continue
        
        compiled_file = compile_submission(submission_path)

        if compiled_file is None:
            print(f"Compilation failed for: {submission_path}")
            continue
        
        valid_submissions += 1
        submission_skipped = False
        print(f"Running tests for: {submission_path}")
        
        passed_cases = 0
        failed_cases = 0
        start_time = time.time()

        for i in range(1, total_cases + 1):
            input_file = os.path.join(test_cases_folder, f"input_{i}.txt")
            output_file = os.path.join(test_cases_folder, f"output_{i}.txt")
            
            if os.path.exists(input_file) and os.path.exists(output_file):
                test_start_time = time.time()
                is_correct = run_test(compiled_file, input_file, output_file)
                test_end_time = time.time()
                test_time = test_end_time - test_start_time
                if test_time >= 10:
                    print(f"Test for {submission_path} took {test_time} seconds. Skipping this submission.")
                    submission_skipped = True
                    break
                
                if is_correct:
                    passed_cases += 1
                else:
                    failed_cases += 1

        end_time = time.time()
        time_taken = end_time - start_time

        if submission_skipped:
            print(f"Skipping submission: {submission_path}")
            valid_submissions -= 1
            continue

        total_passed_cases += passed_cases
        total_failed_cases += failed_cases

        print(f"Results for {submission_path}: Passed {passed_cases}/{total_cases}, Failed {failed_cases}")

        # Check if the limit of 20 submissions has been reached
        if valid_submissions >= max_submissions:
            print("Reached limit of 20 submissions. Moving on to next submission.")
            break
            
    
    # Log the benchmark results    

    print(f"Total passed cases: {total_passed_cases}")
    print(f"Total failed cases: {total_failed_cases}")

    average_passed_cases = total_passed_cases / valid_submissions
    average_failed_cases = total_failed_cases / valid_submissions

    log_benchmark(problem_id, total_submissions, valid_submissions, total_cases, average_passed_cases, average_failed_cases)

def main():
    parser = argparse.ArgumentParser(description="Automation script for code testing and benchmarking")
    parser.add_argument('--problem_id', type=str, required=True, help="Problem ID to process")
    args = parser.parse_args()

    # Load problem description (assumed to be in HTML files)
    problem_description_file = f"problem_descriptions/{args.problem_id}.html"
    with open(problem_description_file, 'r') as file:
        problem_description = file.read()

    # Generate and save test cases using LLM (Ollama)
    generate_and_save_test_cases(args.problem_id, problem_description)

    # Load metadata and test human-generated submissions
    # problem_id = 'p00360'
    submission_metadata = load_problem_metadata(args.problem_id)
    test_submissions(args.problem_id, submission_metadata)

if __name__ == "__main__":
    main()
