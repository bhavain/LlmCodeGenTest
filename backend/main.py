from fastapi import FastAPI
from api.problems import router as problems_router
from api.benchmark import router as benchmark_router
from api.submissions import router as submissions_router
from api.testcases import router as testcases_router

app = FastAPI(title="TestMaker API")

# Include routes
app.include_router(problems_router, prefix="/problems", tags=["Problems"])
app.include_router(benchmark_router, prefix="/benchmark", tags=["Benchmark"])
app.include_router(submissions_router, prefix="/submissions", tags=["Submissions"])
app.include_router(testcases_router, prefix="/testcases", tags=["Test Cases"])

@app.get("/")
def read_root():
    return {"message": "Welcome to TestMaker API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
