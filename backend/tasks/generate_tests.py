from database.models import SessionLocal, TestCase, Problem
from llm.generate_tests import generate_test_cases
from tasks.celery_config import celery_app
from api.problems import get_problem

def generate_and_save_tests_for_problem(generation_id, problem_id):
    """Generate test cases for a single problem asynchronously."""
    if not problem_id:
        return {"error": f"Problem {problem_id} not found"}
    
    session = SessionLocal()
    problem = get_problem(problem_id, session)
    if not problem:
        return {"error": f"Problem {problem_id} not found"}

    test_cases = generate_test_cases(problem.description)

    for case in test_cases:
        test_case = TestCase(
            generation_id=generation_id,
            problem_id=problem_id,
            input_data=case["input"],
            expected_output=case["expected_output"]
        )
        session.add(test_case)
    session.commit()
    session.close()

    return {"message": f"Test cases generated for {problem_id}"}

@celery_app.task
def generate_and_save_tests_for_problems(generation_id, problem_ids):
    """Triggers test case generation for multiple problems in parallel."""
    [generate_and_save_tests_for_problem(generation_id, pid) for pid in problem_ids]
    return {"message": "Test generation started"}
