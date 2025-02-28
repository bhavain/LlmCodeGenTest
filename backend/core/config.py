from dotenv import load_dotenv
import os

# Load environment variables from the correct .env file
# Navigate to the project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  
ENV_FILE = os.path.join(BASE_DIR, "docker.env" if os.getenv("DOCKER_ENV") else ".env")

load_dotenv(ENV_FILE)