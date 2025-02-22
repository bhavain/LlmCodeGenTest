🚀 **TestMaker - AI-powered Test Generation & Benchmarking**

## Overview

TestMaker is an AI-powered test generation and benchmarking tool designed to generate test cases using LLMs (LangChain) and evaluate C++ submissions against these cases. The system supports multi-problem benchmarking, parallel execution, and performance tracking.

## Features

- Generate test cases for multiple problems using LLMs (LangChain).
- Run benchmarking for C++ submissions against generated test cases.
- Utilize FastAPI as the backend API.
- Next.js frontend for managing tests and viewing results.
- Celery for asynchronous task execution.
- PostgreSQL for structured data storage.
- Redis as a message broker for Celery.
- Dockerized for seamless deployment.

## Tech Stack

- **Backend:** FastAPI, Python, SQLAlchemy, Celery
- **Frontend:** Next.js, TypeScript
- **Database:** PostgreSQL
- **Broker:** Redis
- **Containerization:** Docker, Docker Compose
- **Task Queue:** Celery
- **AI Integration:** LangChain

## Project Structure

```
TestMaker/
│── backend/
│   ├── api/                 # FastAPI routes

│   ├── tasks/               # Celery tasks
│   ├── database/
│   │   ├── db_init.py       # Database initialization
│   │   ├── insert_data.py   # Data insertion script
│   |   ├── models.py       # Database models
│   ├── main.py              # FastAPI entry point
│── frontend/
│   ├── pages/               # Next.js pages
│   ├── components/          # Reusable UI components
│   ├── public/              # Static assets
│── docker-compose.yml       # Docker setup
│── Dockerfile               # Backend & frontend containerization
│── docker.env              # Environment variables for Docker
│── local.env               # Environment variables for local development
```

## Setup Instructions

### **1. Clone the repository**

```bash
git clone https://github.com/yourusername/TestMaker.git
cd TestMaker
```

### **2. Set up environment variables**

Create `local.env` for local development:

```ini
DATABASE_URL=postgresql://testmaker_user:password@localhost/testmaker
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
DOCKER_ENV=false
```

Create `docker.env` for Docker deployment:

```ini
DATABASE_URL=postgresql://testmaker_user:password@db/testmaker
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
DOCKER_ENV=true
```

### **3. Run Locally (Without Docker)**

#### **Start Backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

#### **Start Frontend**

```bash
cd frontend
npm install
npm run dev
```

### **4. Run with Docker**

```bash
docker-compose up --build
```

## API Endpoints

### **Test Cases**

- `POST /testcases/generate` – Generate test cases for a single/multiple problems.
- `GET /testcases/{problem_id}` – Fetch test cases for a problem.

### **Benchmarking**

- `POST /benchmark` – Run submissions against generated test cases.
- `GET /results/{generation_id}` – Fetch benchmark results.

## Deployment

- Use `docker-compose` for easy deployment on a cloud VM.
- CI/CD pipeline setup with GitHub Actions (optional).

## Future Enhancements

- Implement real-time progress tracking for test execution.
- Improve UI for better test selection and visualization.
- Support for additional programming languages.
- Deploy on AWS with Kubernetes support.

## License

This project is licensed under [MIT License](LICENSE).

---


