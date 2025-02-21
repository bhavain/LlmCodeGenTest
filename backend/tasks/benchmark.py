import os
import subprocess
from tasks.celery_config import celery_app
from database.models import SessionLocal, TestCase, BenchmarkResult, SubmissionMetadata
from tasks.generate_tests import generate_and_save_tests_for_problems
import time
import datetime
import lizard
# from multiprocessing import Pool, TimeoutError
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEM_SUBMISSIONS_DIR = os.path.join(BASE_DIR, "../database/problem_submissions")

# Compile and test submission
def compile_submission(submission_file):
    output_file = submission_file.replace(".cpp", ".out")
    compile_cmd = f"g++ {submission_file} -o {output_file}"
    
    try:
        subprocess.run(compile_cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file
    except subprocess.CalledProcessError:
        return None

def compile_all_submissions(problem_id, submission_ids):
    """Compile all submissions for a given problem in parallel."""

    problem_submissions_folder = f"{PROBLEM_SUBMISSIONS_DIR}/{problem_id}/"

    # Parallelize compilation
    submission_paths = [
        os.path.join(problem_submissions_folder, f"{sid}.cpp")
        for sid in submission_ids
        if os.path.exists(os.path.join(problem_submissions_folder, f"{sid}.cpp"))
    ]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        compiled_files = executor.map(compile_submission, submission_paths)
    return compiled_files

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

def run_test(compiled_file, input_data, expected_output):
    """
    Runs the compiled program using the input from input_file and compares
    the output with expected_output_file. Includes a timeout of 10 seconds.
    If execution exceeds 10 seconds, the test is marked as failed and skipped.
    """
    try:
        timeout = 10
        # Run the compiled file with the input and capture the output, adding a timeout of 10 seconds
        result = subprocess.run([compiled_file], input=input_data, text=True, timeout=timeout, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = result.stdout.strip()

        expected_output = expected_output.strip()

        # # print both the output and expected output
        # logging.info(f"Output: {output}")
        # logging.info(f"Expected: {expected_output}")

        normalized_output = normalize_output(output)
        normalized_expected = normalize_output(expected_output)

        # logging.info(f"Input: {input_data}")
        # logging.info(f"Normalized Output: {normalized_output}")
        # logging.info(f"Normalized Expected: {normalized_expected}")
        # return output == expected_output
        return normalized_output == normalized_expected

    except subprocess.TimeoutExpired:
        # TimeoutExpired will be raised if the test exceeds the timeout
        logging.warning(f"Test for {compiled_file} timed out after {timeout} seconds. Skipping this test.")
        return False
    
    except subprocess.CalledProcessError:
        # CalledProcessError will be raised if the program returns a non-zero exit code
        return False

def run_all_tests(compiled_file, test_cases, timeout=10):
    """Runs all test cases using ThreadPoolExecutor instead of multiprocessing."""
    results = []

    def run_single_test(test_case):
        """Executes a single test case."""
        try:
            return run_test(compiled_file, test_case["input_data"], test_case["expected_output"])
        except TimeoutError:
            logging.warning(f"Test case {test_case['test_case_id']} timed out. Skipping submission {compiled_file}.")
            return "timeout"

    # ✅ Use ThreadPoolExecutor instead of multiprocessing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_results = {executor.submit(run_single_test, test_case): test_case for test_case in test_cases}

        for future in future_results:
            try:
                results.append(future.result(timeout=timeout))
            except TimeoutError:
                logging.warning(f"Test case {future_results[future]['test_case_id']} timed out.")
                return "timeout"  # Skip this submission

    return results  # Return results if all tests pass within timeout


@celery_app.task
def run_multiple_submissions(generation_id, problem_id, submission_ids, test_cases):
    """Runs multiple submissions for a given problem in parallel."""

    valid_submissions = 0
    total_passed_cases = 0
    total_failed_cases = 0
    total_case_complexity = 0
    total_lines_of_code = 0
    total_token_count = 0
    max_submissions = 20
    total_cases = len(test_cases)


    compiled_files = compile_all_submissions(problem_id, submission_ids)

    # Filter out None values (failed compilations)
    compiled_files = [file for file in compiled_files if file is not None]

    for compiled_file in compiled_files:
        # print(f"Running tests for: {compiled_file}")
        
        # Parallelize test execution
        results = run_all_tests(compiled_file, test_cases)

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

    session = SessionLocal()

    # Log aggregated results
    average_passed_cases = round(total_passed_cases / valid_submissions)
    average_failed_cases = round(total_failed_cases / valid_submissions)
    average_case_complexity = round(total_case_complexity / valid_submissions)
    average_lines_of_code = round(total_lines_of_code / valid_submissions)
    average_token_count = round(total_token_count / valid_submissions)
    created_at=datetime.datetime.utcnow().isoformat()

    # generation_id,total_submissions,average_case_complexity,average_lines_of_code,average_token_count,
    # total_cases,average_passed_cases,average_failed_cases
    benchmark = BenchmarkResult(
        generation_id=generation_id,
        problem_id=problem_id,
        total_submissions=valid_submissions,
        total_cases=total_cases,
        average_passed_cases=average_passed_cases,
        average_failed_cases=average_failed_cases,
        average_case_complexity=average_case_complexity,
        average_lines_of_code=average_lines_of_code,
        average_token_count=average_token_count,
        created_at=created_at
    )
    session.add(benchmark)

    session.commit()
    session.close()

    return {"message": f"Completed execution for problem id {problem_id} - {len(submission_ids)} submissions"}


@celery_app.task
def generate_tests_and_run_all_problems_task(generation_id, problem_ids):
    """Generates tests and runs 20 submissions of all given problems against respective test cases."""
    

    # after all test cases are generated, run 20 submissions for each problem or as test cases are available for each problem start running submissions against them

    generate_task = generate_and_save_tests_for_problems(generation_id, problem_ids)

    session = SessionLocal()
    # run 30 submissions for each problem

    for problem_id in problem_ids:
        # TODO: Randomly pick {num} submissions
        submissions = session.query(SubmissionMetadata).filter(SubmissionMetadata.problem_id == problem_id).limit(30).all()
        test_cases = session.query(TestCase).filter(TestCase.generation_id == generation_id, TestCase.problem_id == problem_id).all()

         # ✅ Extract test case data in JSON-safe format
        json_test_cases = [
            {"test_case_id": tc.id, "input_data": tc.input_data, "expected_output": tc.expected_output}
            for tc in test_cases if tc.problem_id == problem_id
        ]
    
        if not submissions:
            return {"message": "No submissions found for this problem"}

        submission_ids = [sub.submission_id for sub in submissions]
        run_task = run_multiple_submissions.delay(generation_id, problem_id, submission_ids, json_test_cases)

    session.commit()
    session.close()

    # return {"task_id": task.id, "message": f"Started execution for {len(submission_ids)} submissions"}



    return {"message": f"Started generation and execution for {len(problem_ids)} problems"}