from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
import core.config
from api.problems import router as problems_router
from api.benchmark import router as benchmark_router
from api.submissions import router as submissions_router
from api.testcases import router as testcases_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TestMaker API")

IS_DOCKER = os.getenv("DOCKER_ENV", "false") == "true"

origins = ["http://localhost", "http://localhost:3000", "http://localhost:8000"]

if IS_DOCKER:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect if running inside Docker

# Dynamically Set Frontend Path
if IS_DOCKER:
    FRONTEND_PATH = "/app/frontend/out"  # Docker path
else:
    FRONTEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/out"))  # Local path

# Serve Next.js frontend as static files
if os.path.exists(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
else:
    print(f"⚠️ Frontend build not found at: {FRONTEND_PATH}")

# Include routes
app.include_router(problems_router, prefix="/problems", tags=["Problems"])
app.include_router(benchmark_router, prefix="/benchmark", tags=["Benchmark"])
app.include_router(submissions_router, prefix="/submissions", tags=["Submissions"])
app.include_router(testcases_router, prefix="/testcases", tags=["Test Cases"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
