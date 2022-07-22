import os
import time
from labelbox import Client
from google.cloud import storage
from source_code.errors import MissingEnvironmentVariableException
from labelbox.data.serialization import LBV1Converter

def get_gcs_client(google_project: str) -> storage.Client:
    """ Initiates a Google Cloud client object
    Returns:
        google cloud storage client
    """
    if not google_project:
        raise MissingEnvironmentVariableException(f"Must set google_project env var")
    return storage.Client(project=google_project)

def get_lb_client(lb_api_key) -> Client:
    """ Initiates a Labelbox client object
    Returns:
         labelbox client
    """  
    return Client(api_key=lb_api_key, endpoint='https://api.labelbox.com/_gql', enable_experimental=True)

def create_gcs_key(model_run_id: str) -> str:
    """ Utility for creating a gcs key for the etl job
    Args:
        job_name    :       The name of the job (should be named after the etl)
    Returns:
        gcs key for the jsonl file.
    """
    nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    return f'etl/{model_run_id}/{nowgmt}.jsonl'

def env_vars(value):
    """ Pulls envionment variables in from Cloud Functions
    Args:
        value       :       String representing the environment variable to-be-pulled
    """
    return os.environ.get(value, '')

def get_labels_for_model_run(client: Client, model_run_id: str, media_type: str, strip_subclasses: bool:
    """ Exports all labels from a model run
    Args:
        client          :       Labelbox client used for fetching labels
        model_run_id    :       model run to fetch labels for
        media_type      :       Should either be "image" or "text" string
    Returns:
        LabelGenerator with labels to-be-converted into vertex syntax    
    """
    print("Initiating Label Export")
    model_run = client.get_model_run(model_run_id)
    json_labels = model_run.export_labels(download=True)
    print("Label Export Complete")
    for row in json_labels:
        if media_type is not None:
            row['media_type'] = media_type
        if strip_subclasses:
            # Strip subclasses for tools
            for annotation in row['Label']['objects']:
                if 'classifications' in annotation:
                    del annotation['classifications']
            # Strip subclasses for classifications
            for annotation in row['Label']['classifications']:
                if 'answer' in annotation:
                    if 'classifications' in annotation['answer']:
                        del annotation['answer']['classifications']
                if 'answers' in annotation:
                    for answer in annotation['answers']:
                        if 'classifications' in answer:
                            del answer['classifications']
    return LBV1Converter.deserialize(json_labels)
