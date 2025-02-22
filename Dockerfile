# Start from the latest Ubuntu image
FROM ubuntu:latest

# Update apt repositories and install essential tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential

# Using NodeSource to ensure we get the latest Node.js version
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# Install Python, pip, and create a virtual environment
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment and install dependencies
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements first (to leverage Docker caching)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire backend
COPY backend/ /app/

# Setup frontend
COPY frontend/ /app/frontend/
RUN cd /app/frontend && npm install && npm run build

# Create a non-root user
RUN useradd -m celery_user
USER celery_user

# Expose port
EXPOSE 8000

# Serve API & Frontend
CMD ["/app/venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
