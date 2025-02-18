import os
import pandas as pd
from sqlalchemy.orm import Session
from models import Problem, SubmissionMetadata, Base, engine, SessionLocal
import multiprocessing

# Initialize DB
Base.metadata.create_all(bind=engine)

def clean_value(value):
    """Convert NaN to None for safe database insertion."""
    if pd.isna(value):  # Checks for NaN, None, empty values
        return None
    return value

def insert_problems():
    session = SessionLocal()
    
    # Load problem_list.csv
    df = pd.read_csv("problem_list.csv")  # Ensure correct path

    print(f"✅ Loaded problem_list.csv with {len(df)} rows.")
    
    # id,name,dataset,time_limit,memory_limit,rating,tags,complexity
    for _, row in df.iterrows():
        problem = Problem(
            id=row["id"], 
            name=clean_value(row["name"]),
            dataset=clean_value(row.get("dataset")),
            time_limit=clean_value(row.get("time_limit")),
            memory_limit=clean_value(row.get("memory_limit")),
            rating=clean_value(row.get("rating")),
            tags=clean_value(row.get("tags")),
            complexity=clean_value(row.get("complexity")),
        )
        session.add(problem)
        # print(f"Inserted problem: {problem.id} - {problem.name}")

    session.commit()
    session.close()
    print("✅ Problems inserted successfully!")


def process_submission_file(problem_id):
    """Process a single metadata file and insert submissions into the database."""
    session = SessionLocal()  # Each process gets its own DB session

    # Get absolute path of the script's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(BASE_DIR, "problems_metadata", f"{problem_id}.csv")

    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        submissions = []

        for _, row in df.iterrows():

            language = clean_value(row.get("language"))
            status = clean_value(row.get("status"))

            if language and language.lower() == "c++" and status and status.lower() == "accepted":
                submission = SubmissionMetadata(
                    problem_id=problem_id,
                    submission_id=row["submission_id"],
                    user_id=clean_value(row.get("user_id")),
                    date=clean_value(row.get("date")),
                    language=language,
                    original_language=clean_value(row.get("original_language")),
                    filename_ext=clean_value(row.get("filename_ext")),
                    status=status,
                    cpu_time=clean_value(row.get("cpu_time")),
                    memory=clean_value(row.get("memory")),
                    code_size=clean_value(row.get("code_size")),
                    accuracy=clean_value(row.get("accuracy")),
                )
                submissions.append(submission)

        if submissions:
            session.bulk_save_objects(submissions)  # Efficient batch insert
            session.commit()
            print(f"✅ Inserted {len(submissions)} submissions for problem {problem_id}")
        else:
            print(f"⚠️ No valid submissions for problem {problem_id}")

    else:
        print(f"❌ Metadata file for problem {problem_id} not found.")

    session.close()  # Close session for this process

def insert_submissions():
    """Run multiprocessing to insert submissions for all problems."""
    problems = [f.replace(".csv", "") for f in os.listdir("problems_metadata") if f.endswith(".csv")]

    print(f"🛠️ Starting multiprocessing for {len(problems)} problems...")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_submission_file, problems)

    print("✅ Submissions Metadata inserted successfully!")



if __name__ == "__main__":
    # insert_problems()
    insert_submissions()
