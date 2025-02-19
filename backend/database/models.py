from sqlalchemy import Column, Integer, String, Float, BigInteger, create_engine, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# PostgreSQL Connection URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://testmaker_user:RubberDuck%4029@localhost/testmaker")

Base = declarative_base()

# id,name,dataset,time_limit,memory_limit,rating,tags,complexity
class Problem(Base):
    __tablename__ = "problems"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    dataset = Column(String, nullable=True)
    time_limit = Column(BigInteger, nullable=True)
    memory_limit = Column(BigInteger, nullable=True)
    rating = Column(Integer, nullable=True)
    tags = Column(String, nullable=True)
    complexity = Column(Float, nullable=True)

# submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy
class SubmissionMetadata(Base):
    __tablename__ = "submission_metadata"
    submission_id = Column(String, primary_key=True, index=True)
    problem_id = Column(String, ForeignKey("problems.id"), nullable=False)
    user_id = Column(String, nullable=True)
    date = Column(String, nullable=True)
    language = Column(String, nullable=False)
    original_language = Column(String, nullable=False)
    filename_ext = Column(String, nullable=True)
    status = Column(String, nullable=True)
    cpu_time = Column(Float, nullable=True)
    memory = Column(Float, nullable=True)
    code_size = Column(Integer, nullable=True)
    accuracy = Column(String, nullable=True)

class TestGeneration(Base):
    """Tracks a test generation session for one or more problems."""
    __tablename__ = "test_generations"

    generation_id = Column(String, primary_key=True, index=True)  # Unique test generation session
    problems = Column(Text, nullable=False)  # Comma-separated list of problem IDs
    status = Column(String, default="pending")  # pending, completed
    created_at = Column(String, nullable=False)

class TestCase(Base):
    __tablename__ = "test_cases"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    generation_id = Column(String, ForeignKey("test_generations.generation_id"), nullable=False)
    problem_id = Column(String, ForeignKey("problems.id"), nullable=False)
    input_data = Column(Text, nullable=True)
    expected_output = Column(Text, nullable=True)

# generation_id,problem_id,total_submissions,average_case_complexity,average_lines_of_code,average_token_count,
# total_cases,average_passed_cases,average_failed_cases
class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    generation_id = Column(String, ForeignKey("test_generations.generation_id"), nullable=False)
    problem_id = Column(String, ForeignKey("problems.id"), nullable=False)
    total_cases = Column(Integer, nullable=False)
    passed_cases = Column(Integer, nullable=False)
    failed_cases = Column(Integer, nullable=False)
    average_case_complexity = Column(Float, nullable=False)
    average_lines_of_code = Column(Float, nullable=False)
    average_token_count = Column(Float, nullable=False)
    total_submissions = Column(Integer, nullable=False)
    created_at = Column(String, nullable=False)

# Initialize Database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
