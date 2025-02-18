from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.models import SessionLocal

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()