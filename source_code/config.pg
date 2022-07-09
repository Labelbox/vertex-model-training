import os
from labelbox import Client
from google.cloud import storage
import time
from source_code.errors import MissingEnvironmentVariableException

def get_gcs_client() -> storage.Client:
    """
    Returns:
        google cloud storage client.
    """
    google_project = env_vars("google_project")
    if not google_project:
        raise MissingEnvironmentVariableException(f"Must set google_project env var")
    return storage.Client(project=google_project)

def get_lb_client() -> Client:
    """
    Returns:
         labelbox client.
    """
    api_key = env_vars("api_key")
    if not api_key:
        raise MissingEnvironmentVariableException(f"Must set api_key env var")    
    return Client(api_key=api_key, endpoint='https://api.labelbox.com/_gql', enable_experimental=True)

def create_gcs_key(model_run_id: str) -> str:
    """
    Utility for creating a gcs key for etl jobs
    Args:
        job_name: The name of the job (should be named after the etl)
    Returns:
        gcs key for the jsonl file.
    """
    nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    return f'etl/{model_run_id}/{nowgmt}.jsonl'

def env_vars(value):
    return os.environ.get(value, '')
