import os
from labelbox import Client
from google.cloud import storage
import time
from source_code.errors import MissingEnvironmentVariableException

def get_gcs_client(google_project: str) -> storage.Client:
    """
    Returns:
        google cloud storage client.
    """
    google_project = os.environ.get(google_project)
    if not google_project:
        raise MissingEnvironmentVariableException(f"Must set GOOGLE_PROJECT env var")
    return storage.Client(project=google_project)

def get_lb_client(api_key: str) -> Client:
    """
    Returns:
         labelbox client.
    """
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