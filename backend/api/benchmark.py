from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.models import SessionLocal, SubmissionMetadata, Problem
from tasks.benchmark import generate_tests_and_run_all_problems
from pydantic import BaseModel
from api.testcases import create_new_test_generation_id

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class BenchmarkRequest(BaseModel):
    problem_ids: list[str]

@router.post("/")
def generate_tests_and_run_all_problems(request: BenchmarkRequest, db: Session = Depends(get_db)):
    """Generates tests and runs 20 submissions of all given problems against respective test cases."""

    problem_ids = request.problem_ids
    if not problem_ids or not isinstance(problem_ids, list):
        return {"error": "Invalid input. Expected a list of problem IDs."}
    
    # TODO: Make this more efficient, is this necessary?
    # Verify if all problems exist in the database
    problems = db.query(Problem).filter(Problem.id.in_(problem_ids)).all()
    if len(problems) != len(problem_ids):
        return {"error": "One or more problems not found"}
    
    # create a new test generation session
    generation_details = create_new_test_generation_id(problem_ids)

    if not generation_details:
        return {"error": "Failed to create test generation session"}

    task = generate_tests_and_run_all_problems.delay(generation_details["generation_id"], problem_ids)
    generation_details["task_id"] = task.id

    return generation_details
