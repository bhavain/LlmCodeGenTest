from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.models import SessionLocal, TestCase, TestGeneration, Problem
from tasks.generate_tests import generate_and_save_tests_for_problems
import uuid
import datetime
from pydantic import BaseModel

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Define Pydantic model to enforce correct structure
class GenerateRequest(BaseModel):
    problem_ids: list[str]

@router.post("/generate")
def generate_tests(request: GenerateRequest, db: Session = Depends(get_db)):
    """Triggers LLM to generate test cases for a given problem or multiple problems."""

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

    task = generate_and_save_tests_for_problems.delay(generation_details["generation_id"], problem_ids)
    generation_details["task_id"] = task.id

    return generation_details

@router.get("/{problem_id}")
def get_test_cases(problem_id: str, db: Session = Depends(get_db)):
    """Fetches generated test cases for a given problem."""
    test_cases = db.query(TestCase).filter(TestCase.problem_id == problem_id).all()
    return test_cases if test_cases else {"message": "No test cases found"}


def create_new_test_generation_id(problem_ids: list[str]):
    """Starts a new test generation session for multiple problems."""
    db = SessionLocal()
    generation_id = str(uuid.uuid4())  # Generate unique generation ID
    created_at = datetime.datetime.utcnow().isoformat()

    # Convert list of problems to a comma-separated string
    problem_list = ",".join(problem_ids)

    session = TestGeneration(generation_id=generation_id, problems=problem_list, status="pending", created_at=created_at)
    db.add(session)
    db.commit()
    db.close()

    return {"generation_id": generation_id, "message": "Test generation session started", "problems": problem_ids}

@router.get("generation_details/{generation_id}")
def get_test_generation_details(generation_id: str, db: Session = Depends(get_db)):
    """Fetch details of a test generation session."""
    session = db.query(TestGeneration).filter(TestGeneration.generation_id == generation_id).first()
    if not session:
        return {"error": "Test generation session not found"}

    # Convert stored problem list back to array
    problem_list = session.problems.split(",")

    return {
        "generation_id": session.generation_id,
        "problems": problem_list,
        "status": session.status,
        "created_at": session.created_at
    }