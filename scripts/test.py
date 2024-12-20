import os
import subprocess
import pandas as pd
import ollama
import argparse
import time
import json
import lizard
from multiprocessing import Pool, TimeoutError
import random

import logging
from jsonschema import validate, ValidationError
from bs4 import BeautifulSoup

# Define a JSON schema for test cases
TEST_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "test_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input": {"anyOf": [
                        {"type": "string"},
                        {"type": "array"},
                        {"type": "object"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"}
                    ]},
                    "expected_output": {"anyOf": [
                        {"type": "string"},
                        {"type": "array"},
                        {"type": "object"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"}
                    ]}
                },
                "required": ["input", "expected_output"]
            }
        }
    },
    "required": ["test_cases"]
}

# Configure logging
logging.basicConfig(
    filename="automation.log",  # Log file name
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

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
        subprocess.run(compile_cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file
    except subprocess.CalledProcessError:
        return None
    
def normalize_output(output):
    """
    Normalize output to a single line of space-separated values for comparison.
    Handles multiline outputs and ensures consistent formatting.

    Args:
        output (str): The raw output to normalize.

    Returns:
        str: A normalized single-line string for comparison.
    """
    # Split into lines, strip each line, and join with a space
    return " ".join(line.strip() for line in output.strip().splitlines() if line.strip())

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
            result = subprocess.run([compiled_file], input=input_data.read(), text=True, timeout=timeout, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output = result.stdout.strip()

        with open(expected_output_file, 'r') as expected_output_file_content:
            expected_output = expected_output_file_content.read().strip()

        # # print both the output and expected output
        # logging.info(f"Output: {output}")
        # logging.info(f"Expected: {expected_output}")

        normalized_output = normalize_output(output)
        normalized_expected = normalize_output(expected_output)


        logging.info(f"Normalized Output: {normalized_output}")
        logging.info(f"Normalized Expected: {normalized_expected}")
        # return output == expected_output
        return normalized_output == normalized_expected

    except subprocess.TimeoutExpired:
        # TimeoutExpired will be raised if the test exceeds the timeout
        logging.warning(f"Test for {compiled_file} timed out after {timeout} seconds. Skipping this test.")
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

def strip_html_tags(html):
    """
    Strips HTML tags from a given string.

    Args:
    html (str): String containing HTML content.

    Returns:
    str: Text content without HTML tags.
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def extract_format_from_description(problem_description):
    """
    Extract the input and output format from a structured problem description.
    Returns the input format, output format, and constraints as strings.
    """
    input_format = ""
    output_format = ""
    constraints = ""

    if "Input" in problem_description:
        input_start = problem_description.find("Input")
        output_start = problem_description.find("Output")
        sample_input_start = problem_description.find("Sample Input")
        # constraints_start = problem_description.find("Constraints")

        input_format = problem_description[input_start:output_start]
        output_format = problem_description[output_start:sample_input_start]
        # constraints = problem_description[constraints_start:].strip()

        # Strip HTML tags and extra whitespace
        input_format = strip_html_tags(input_format)
        output_format = strip_html_tags(output_format)

    return input_format, output_format, constraints

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
    logging.info(f"Generating {remaining_test_cases} new test cases for problem {problem_id}...")

    # Extract input/output format and constraints from the problem description
    input_format, output_format, constraints = extract_format_from_description(problem_description)

    # logging.info(f"Input Format: {input_format}")
    # logging.info(f"Output Format: {output_format}")
    # logging.info(f"Constraints: {constraints}")


    # Prompt Ollama to generate the remaining test cases
    response_old = ollama.chat(model="llama3.1", messages=[
        {
            "role": "user", 
            "content": f"""

            You are a coding and testing expert. Generate {remaining_test_cases} diverse test cases in JSON format for the following programming problem.
            
            Problem Description:
            {problem_description}

            Input Format:
            {input_format}

            Output Format:
            {output_format}
            
            The test cases should cover:
            1. **Basic Cases**: Simple scenarios to test basic functionality.

            Ensure the JSON output is strictly in the following format:
            {{
                "test_cases": [
                    {{
                        "input": "<formatted input>",
                        "expected_output": "<formatted output>"
                    }},
                    ...
                ]
            }}

            Guidelines:
            - Inputs and outputs should be valid and free of syntax errors.
            - Clearly separate each input-output pair.
            - Do not include any explanations or additional text.

            Validate the JSON before providing the response. Ensure no missing or incomplete data.
            """
        }
    ])

    # Prompt Ollama to generate the remaining test cases
    response = ollama.chat(model="llama3.1", messages=[
        {
            "role": "system",
            "content": "You are a JSON generation assistant specializing in creating programming test cases."
        },
        {
            "role": "user", 
            "content": f"""

            Generate {remaining_test_cases} test cases for the following problem logically.
            
            Problem Description:
            {problem_description}

            Input Format:
            {input_format}

            Output Format:
            {output_format}

            Step 1: Understand the input/output structure.
            Step 2: Identify key cases
            Step 3: Generate inputs.
            Step 4: Compute outputs logically.
            Step 5: Compile the results into JSON format.

            Ensure the JSON output is strictly in the following format:
            {{
                "test_cases": [
                    {{
                        "input": "<formatted input>",
                        "expected_output": "<formatted output>"
                    }},
                    ...
                ]
            }}

            Guidelines:
            - Inputs and outputs should be valid and free of syntax errors.
            - Clearly separate each input-output pair.
            - Do not include any explanations or additional text.

            Validate and ensure the test cases are logically consistent.
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
    
    logging.info(f"{remaining_test_cases} new test cases saved for problem {problem_id} in {problem_test_dir}")
    # return inputs, outputs

def extract_input_output_json(test_cases_json):
    inputs = []
    outputs = []

    # Remove extra text and isolate JSON
    json_start = test_cases_json.find('{')
    json_end = test_cases_json.rfind('}')
    if json_start != -1 and json_end != -1:
        clean_json = test_cases_json[json_start:json_end + 1]
    else:
        logging.error("No valid JSON found.")
        return inputs, outputs

    # Parse and validate JSON
    try:
        # logging.info(f"Clean JSON: {clean_json}")
        test_cases = json.loads(clean_json)
        # logging.info(f"Test cases: {test_cases}")
        # test_cases = {'test_cases': [{'input': ['1819', '2003', '876', '2840', '1723', '1673', '3776', '2848', '1592', '922'], 'expected_output': ['3776', '2848', '2840']}, {'input': ['100', '200', '300', '400', '500', '600', '700', '800', '900', '900'], 'expected_output': ['900', '900', '800']}, {'input': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 'expected_output': ['10', '9', '8']}, {'input': ['10000', '9999', '9998', '9997', '9996', '9995', '9994', '9993', '9992', '9991'], 'expected_output': ['10000', '9999', '9998']}, {'input': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'expected_output': ['9', '8', '7']}, {'input': ['20000', '10001', '5000', '3000', '2500', '1000', '500', '100', '50', '10'], 'expected_output': ['20000', '10001', '5000']}, {'input': [900, 900, 800, 700, 600, 500, 400, 300, 200, 100], 'expected_output': ['900', '900', '800']}, {'input': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'expected_output': ['100', '90', '80']}]}
        validate(instance=test_cases, schema=TEST_CASE_SCHEMA)  # Enforce schema

        for test_case in test_cases["test_cases"]:
            input_value = test_case["input"]
            expected_output = test_case["expected_output"]

            # logging.info(f"Input: {input_value}")
            # logging.info(f"Expected Output: {expected_output}")

            # Normalize input and output for saving
            normalized_input = normalize_data(input_value)
            normalized_output = normalize_data(expected_output)

            # logging.info(f"Normalized Input: {normalized_input}")
            # logging.info(f"Normalized Output: {normalized_output}")

            inputs.append(normalized_input)
            outputs.append(normalized_output)

    except ValidationError as e:
        logging.error(f"JSON does not match schema: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")

    return inputs, outputs

def normalize_data(value):
    """
    Convert the input/output value to a normalized string representation for saving.
    Handles different types (e.g., array, object, string).
    """
    if isinstance(value, list):
        return " ".join(map(str, value))  # Join list elements with a space
    if isinstance(value, dict):
        return json.dumps(value)  # Convert to JSON string
    elif isinstance(value, (int, float, bool, type(None))):
        return str(value)  # Convert to string
    elif isinstance(value, str):
        return value.strip()
    else:
        return str(value)  # Fallback for unknown types

def extract_input_output_json_2(test_cases_json):
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
                logging.error(f"Error parsing JSON: {e}")
        else:
            logging.error("Error: JSON format is incomplete.")
    else:
        logging.error("Error: No valid JSON found in response.")

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
            logging.warning(f"Submission file not found: {submission_path}")
            continue
        
        compiled_file = compile_submission(submission_path)

        if compiled_file is None:
            logging.error(f"Compilation failed for: {submission_path}")
            continue
        
        valid_submissions += 1
        submission_skipped = False
        logging.info(f"Running tests for: {submission_path}")
        
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
                    logging.warning(f"Test for {submission_path} took {test_time} seconds. Skipping this submission.")
                    submission_skipped = True
                    break
                
                if is_correct:
                    passed_cases += 1
                else:
                    failed_cases += 1

        end_time = time.time()
        time_taken = end_time - start_time

        if submission_skipped:
            logging.warning(f"Skipping submission: {submission_path}")
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

        logging.info(f"Results for {submission_path}: Passed {passed_cases}/{total_cases}, Failed {failed_cases}")

        # Check if the limit of 20 submissions has been reached
        if valid_submissions >= max_submissions:
            logging.info("Reached limit of 20 submissions. Moving on to next submission.")
            break
            
    
    # Log the benchmark results    

    # print(f"Total passed cases: {total_passed_cases}")
    # print(f"Total failed cases: {total_failed_cases}")

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

def compile_all_submissions(submission_files):
    with Pool(processes=os.cpu_count()) as pool:
        compiled_files = pool.map(compile_submission, submission_files)
    return compiled_files

def execute_test(compiled_file, test_cases_folder, i):
    input_file = os.path.join(test_cases_folder, f"input_{i}.txt")
    output_file = os.path.join(test_cases_folder, f"output_{i}.txt")
    if os.path.exists(input_file) and os.path.exists(output_file):
        return run_test(compiled_file, input_file, output_file)
    return False  # Mark as failed if files are missing

def run_all_tests(compiled_file, test_cases_folder, total_cases, timeout=10):
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        for i in range(1, total_cases + 1):
            try:
                # Apply async with timeout for each test
                result = pool.apply_async(execute_test, args=(compiled_file, test_cases_folder, i))
                results.append(result.get(timeout))  # Retrieve result with timeout
            except TimeoutError:
                logging.warning(f"Test case {i} timed out. Skipping submission {compiled_file}.")
                pool.terminate()  # Stop all ongoing tasks
                return "timeout"  # Skip this submission
        pool.close()
        pool.join()
    return results  # Return results if all tests pass within timeout

def test_submissions_multi(problem_id, submission_metadata):
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

    # Parallelize compilation
    submission_paths = [
        os.path.join(problem_folder, f"{row['submission_id']}.cpp")
        for _, row in submission_metadata.iterrows()
        if os.path.exists(os.path.join(problem_folder, f"{row['submission_id']}.cpp"))
    ]

    # Randomly pick sample_size submissions
    sample_size = 25
    if len(submission_paths) > sample_size:
        submission_paths = random.sample(submission_paths, sample_size)

    compiled_files = compile_all_submissions(submission_paths)

    # Filter out None values (failed compilations)
    compiled_files = [file for file in compiled_files if file is not None]

    for compiled_file in compiled_files:
        # print(f"Running tests for: {compiled_file}")
        
        # Parallelize test execution
        results = run_all_tests(compiled_file, test_cases_folder, total_cases)

        if results == "timeout":
            logging.info(f"Skipping submission {compiled_file} due to timeout.")
            continue

        # Perform test results analysis if no timeout occurred
        valid_submissions += 1
        passed_cases = sum(1 for res in results if res)
        failed_cases = len(results) - passed_cases

        # Analysis
        submission_path = compiled_file.replace(".out", ".cpp")
        lizard_analysis = lizard.analyze_file(submission_path)
        code_complexity = lizard_analysis.function_list[0].cyclomatic_complexity
        lines_of_code = lizard_analysis.function_list[0].nloc
        token_count = lizard_analysis.function_list[0].token_count

        total_case_complexity += code_complexity
        total_lines_of_code += lines_of_code
        total_token_count += token_count
        total_passed_cases += passed_cases
        total_failed_cases += failed_cases

        logging.info(f"Results for {submission_path}: Passed {passed_cases}/{total_cases}, Failed {failed_cases}")

        if valid_submissions >= max_submissions:
            logging.info("Reached limit of 20 submissions. Moving on.")
            break

    # Log aggregated results
    average_passed_cases = round(total_passed_cases / valid_submissions)
    average_failed_cases = round(total_failed_cases / valid_submissions)
    average_case_complexity = round(total_case_complexity / valid_submissions)
    average_lines_of_code = round(total_lines_of_code / valid_submissions)
    average_token_count = round(total_token_count / valid_submissions)

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
    test_submissions_multi(args.problem_id, submission_metadata)

if __name__ == "__main__":
    main()
