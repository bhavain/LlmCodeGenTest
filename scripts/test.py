import os
import subprocess
import pandas as pd
import ollama
import argparse
import time
import json
import lizard

# Logging function to track benchmarking results
def log_benchmark(benchmark_results):
    log_file = "benchmark_results.csv"
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("problem_id,total_submissions,valid_submissions,average_case_complexity,average_lines_of_code,average_token_count,total_cases,average_passed_cases,average_failed_cases\n")
    with open(log_file, 'a') as f:
        problem_id = benchmark_results["problem_id"]
        total_submissions = benchmark_results["total_submissions"]
        valid_submissions = benchmark_results["valid_submissions"]
        average_case_complexity = benchmark_results["average_case_complexity"]
        average_lines_of_code = benchmark_results["average_lines_of_code"]
        average_token_count = benchmark_results["average_token_count"]
        total_cases = benchmark_results["total_cases"]
        average_passed_cases = benchmark_results["average_passed_cases"]
        average_failed_cases = benchmark_results["average_failed_cases"]
        f.write(f"{problem_id},{total_submissions},{valid_submissions},{average_case_complexity},{average_lines_of_code},{average_token_count},{total_cases},{average_passed_cases},{average_failed_cases}\n")

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

        # # print both the output and expected output
        # print(f"Output: {output}")
        # print(f"Expected: {expected}")

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
    print(f"Generating {remaining_test_cases} new test cases for problem {problem_id}...")

    # Prompt Ollama to generate the remaining test cases
    response = ollama.chat(model="llama3.1", messages=[
        {
            "role": "user", 
            "content": f"""

            You are a code generation expert. Generate {remaining_test_cases} test cases in JSON format for the following problem:
        
            {problem_description}
            
            The format of the JSON output should be:
            {{
                "test_cases": [
                    {{
                        "input": <formatted input here>,
                        "expected_output": <formatted output here>
                    }},
                    ...
                ]
            }}

            Please ensure that the input and output are simple and easy to parse. 
            Do not include any explanations or additional text.
            """
        }
    ])
    
    # Extract input/output from response
    test_cases = response['message']['content']
    # print(test_cases)
    inputs, outputs = extract_input_output_json(test_cases)  # Implement this to parse the test cases
    # return

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

def extract_input_output_json(test_cases_json):
    """
    Parse the JSON-formatted test cases from the Ollama response to extract inputs and outputs.
    JSON format:
    {
        "test_cases": [
            {
                "input": <input value>,
                "expected_output": <output value>
            },
            ...
        ]
    }
    """
    inputs = []
    outputs = []

    # Remove extra text before the JSON starts
    json_start = test_cases_json.find('{')
    if json_start != -1:
        clean_json = test_cases_json[json_start:]
        
        # Ensure the JSON ends correctly
        json_end = clean_json.rfind('}')
        if json_end != -1:
            clean_json = clean_json[:json_end + 1]

            # Parse the JSON response
            try:
                test_cases = json.loads(clean_json)
                
                # Extract inputs and outputs
                for test_case in test_cases.get("test_cases", []):
                    input_value = test_case.get("input")
                    expected_output = test_case.get("expected_output")

                    # print(f"Input: {input_value}")

                    # Ensure input_value is a string, regardless of its original format
                    if isinstance(input_value, list):
                        # Remove surrounding quotes if present
                        # input_value = [str(item).strip("'\"") for item in input_value if item]
                        input_value = [str(item) for item in input_value if str(item).strip()]
                        # print(f"Modified Input: {input_value}")
                        input_value = ' '.join(input_value)                    
                    else:
                        input_value = str(input_value).strip("'\"")

                    # print(f"Modified Input 2: {input_value}")

                    # Ensure expected_output is a string, regardless of its original format
                    if isinstance(expected_output, list):
                        expected_output = [str(item) for item in expected_output if str(item).strip()]
                        expected_output = ' '.join(expected_output)
                    else:
                        expected_output = str(expected_output).strip("'\"")

                    inputs.append(input_value)
                    outputs.append(expected_output)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
        else:
            print("Error: JSON format is incomplete.")
    else:
        print("Error: No valid JSON found in response.")

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
    problem_folder = f"problem_submissions/{problem_id}/"
    test_cases_folder = f"problem_tests_generated/{problem_id}/"

    total_submissions = len(submission_metadata)
    valid_submissions = 0
    total_passed_cases = 0
    total_failed_cases = 0
    total_case_complexity = 0
    total_lines_of_code = 0
    total_token_count = 0
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

        lizard_analysis = lizard.analyze_file(submission_path)
        code_complexity = lizard_analysis.function_list[0].cyclomatic_complexity
        lines_of_code = lizard_analysis.function_list[0].nloc
        token_count = lizard_analysis.function_list[0].token_count

        total_case_complexity += code_complexity
        total_lines_of_code += lines_of_code
        total_token_count += token_count

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
    average_case_complexity = total_case_complexity / valid_submissions
    average_lines_of_code = total_lines_of_code / valid_submissions
    average_token_count = total_token_count / valid_submissions

    benchmarking_results = {
        "problem_id": problem_id,
        "total_submissions": total_submissions,
        "valid_submissions": valid_submissions,
        "average_case_complexity": average_case_complexity,
        "average_lines_of_code": average_lines_of_code,
        "average_token_count": average_token_count,
        "total_cases": total_cases,
        "average_passed_cases": average_passed_cases,
        "average_failed_cases": average_failed_cases,
    }

    log_benchmark(benchmarking_results)

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
