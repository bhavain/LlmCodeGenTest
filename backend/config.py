from dotenv import load_dotenv
import os

# Load environment variables from the correct .env file
ENV_FILE = "docker.env" if os.getenv("DOCKER_ENV") else "local.env"
load_dotenv(ENV_FILE)