from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.models import SessionLocal, Problem
import os

router = APIRouter()

# Define base path for problem descriptions
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # API folder
PROBLEM_DESC_DIR = os.path.join(BASE_DIR, "../database/problem_descriptions")  # Adjust path


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_problems(db: Session = Depends(get_db)):
    return db.query(Problem).all()

@router.get("/{problem_id}")
def get_problem(problem_id: str, db: Session = Depends(get_db)):
    problem = db.query(Problem).filter(Problem.id == problem_id).first()
    if not problem:
        return {"error": "Problem not found"}
    file_path = os.path.join(PROBLEM_DESC_DIR, f"{problem_id}.html")
    if not os.path.exists(file_path):
        return {"error": "Problem description not found"}
    
    with open(file_path, "r") as file:
        description = file.read()

    problem.description = description
    
    return problem
