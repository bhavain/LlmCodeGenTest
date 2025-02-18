from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.models import SessionLocal, SubmissionMetadata

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{problem_id}/submissions_metadata")
def get_submissions_metadata(problem_id: str, db: Session = Depends(get_db)):
    """Fetch all submissions metadata for a given problem"""
    submissions = db.query(SubmissionMetadata).filter(
        SubmissionMetadata.problem_id == problem_id,
    ).all()

    if not submissions:
        return {"error": "No submissions metadata found for this problem"}
    
    return submissions