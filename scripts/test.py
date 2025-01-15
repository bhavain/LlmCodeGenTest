import os
import subprocess
import pandas as pd
import argparse
import time
import lizard
from multiprocessing import Pool, TimeoutError
import random

import logging

from test_cases import generate_and_save_test_cases

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
