def monitor_function(request):
    import json
    import time
    from google.cloud import aiplatform
    from source_code.config import env_vars 
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    time.sleep(300)
    training_job = aiplatform.AutoMLImageTrainingJob.list(filter=f'display_name={request_json["job_name"]}')[0]
    job_state = str(training_job.state)
    completed_states = [
        "PipelineState.PIPELINE_STATE_SUCCEEDED",
        "PipelineState.PIPELINE_STATE_FAILED",
        "PipelineState.PIPELINE_STATE_CANCELLED",
        "PipelineState.PIPELINE_STATE_PAUSED",     
    ]
    if job_state in completed_states:
        print('Training compete, send to inference')
        # requests.post(monitor_url, data=request_bytes)
    else:
        print('Training incomplete, will check again in 5 minutes')
        requests.post(request_json['monitor_url'], data=request_bytes)
    return "Monitor Job"
    

def train_function(request):
    """
    Initates a training job and the monitor cloud function
    """
    import json
    import requests
    from source_code.config import env_vars    
    from source_code.etl import create_vertex_dataset
    from source_code.train import create_autoML_training_job
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    vertex_dataset = create_vertex_dataset(request_json['model_run_id'], request_json['etl_file'])
    requests.post(request_json['monitor_url'], data=request_bytes)    
    vertex_model, vertex_model_id = create_autoML_training_job(model_name, vertex_dataset, model_run_id)
    print(f'Training Job Name: {model_name}')
    print(f'Vertex Model ID: {vertex_model_id}')
    return "Train Job"

def etl_function(request):
    """
    Receives an ETL webhook trigger and returns a vertex dataset
    Environment Variables:
        api_key         :       Labelbox API Key
        gcs_bucket      :       Name of a GCS Bucket to-be-generated
        google_project  :       Name of the google project this Cloud Function is in
    Returns:
        Google Bucket with the data rows from the model run, ready to-be-converted into a Vertex Dataset
        A dictionary where {model_run_id : etl_file}
    """
    import json
    from source_code.config import env_vars, create_gcs_key, get_lb_client, get_gcs_client
    from source_code.etl import etl_job, upload_ndjson_data
    from labelbox import Client
    from google.cloud import storage
    from google.cloud import aiplatform
    import requests
    print("Initiating ETL")
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    model_id = request_json['modelId']
    model_run_id = request_json['modelRunId']
    model_type = request_json['modelType']
    lb_client = get_lb_client()
    bucket = get_gcs_client().create_bucket(env_vars('gcs_bucket'), location = 'US-CENTRAL1')
    json_data = etl_job(lb_client, model_run_id, bucket)
    gcs_key = create_gcs_key(model_run_id)
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    print(f'ETL File: {etl_file}')
    post_dict = {
        "model_run_id" : model_run_id,
        "etl_file" : etl_file,
        "api_key" : env_vars("api_key"),
        "google_project" : env_vars("google_project"),
        "train_url" : env_vars("train_url"),
        "monitor_url" : env_vars("monitor_url"),
        "inference_url" : env_vars("inference_url"),
        "model_name" : env_vars("model_name")
    }
    post_bytes = json.dumps(post_dict).encode('utf-8')
    requests.post(post_dict['train_url'], data=post_bytes)
    print(f"ETL Complete. Training Job Initiated.")
    return "ETL Job"

def model_run(request):
    """
    Reroutes the webhook trigger to the ETL function
    Environment Variables:
        etl_url         :           URL for the ETL Cloud Function Trigger
    """
    import json
    import requests
    from source_code.config import env_vars
    
    string = request.get_data()
    etl_url = env_vars("etl_url")
    requests.post(etl_url, data=string)

    return "Rerouting to ETL"

def models(request):
    """To-be-used in a Google Cloud Function.
    Args:
        model_options           :           A list of model names you want to appear in the Labelbox UI
    """

    model_options = [ ## Input list of model options here
        "image_classification_custom_model"
    ]

    models_dict = {}

    for model in model_options:
        models_dict.update({model : []})

    return models_dict
