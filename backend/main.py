from fastapi import FastAPI
from api.problems import router as problems_router
from api.automation import router as automation_router
from api.results import router as results_router
from api.submissions import router as submissions_router

app = FastAPI(title="TestMaker API")

# Include routes
app.include_router(problems_router, prefix="/problems", tags=["Problems"])
app.include_router(automation_router, prefix="/automation", tags=["Automation"])
app.include_router(results_router, prefix="/results", tags=["Results"])
app.include_router(submissions_router, prefix="/submissions", tags=["Submissions"])

@app.get("/")
def read_root():
    return {"message": "Welcome to TestMaker API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
