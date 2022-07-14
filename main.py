def monitor_function(request):
    from google.cloud import aiplatform
    from source_code.config import env_vars 
    
    job_name = env_vars("job_name")
    
    training_job = aiplatform.AutoMLImageTrainingJob.list(filter=f'display_name={job_name}')[0]
    
    try:
        print('Use parenthesis')
        x = str(training_job.state())
    except:
        print('Dont')   
        x = str(training_job.state)
    
    completed_states = [
        "PipelineState.PIPELINE_STATE_SUCCEEDED",
        "PipelineState.PIPELINE_STATE_FAILED",
        "PipelineState.PIPELINE_STATE_CANCELLED",
        "PipelineState.PIPELINE_STATE_PAUSED",     
    ]
    
    if x in completed_states:
        print('Training compete, send to inference')
    else:
        print('Training incomplete, will check again in X minutes')
    
    return "Monitor Job"
    

def train_function(request):
    import json
    from source_code.config import env_vars    
    from source_code.etl import create_vertex_dataset
    from source_code.train import create_autoML_training_job
    
    request_bytes = request.get_data()
    print(request_bytes)
    
    request_json = json.loads(request_bytes)
    print(request_json)
    
    model_run_id = list(request_json.keys())[0]
    etl_file = list(request_json.values())[0]    
    
    model_name = env_vars('model_name')

    vertex_dataset = create_vertex_dataset(model_run_id, etl_file)
    vertex_model, vertex_model_id = create_autoML_training_job(model_name, vertex_dataset, model_run_id)
    
    print(f'Vertex Model ID: {vertex_model_id}')
    print(f'Vertex Model: {vertex_model}')
    
    return "Train Job"

def etl_function(request):
    """
    Receives an ETL webhook trigger and returns a vertex dataset
    Environment Variables:
        api_key         :       Labelbox API Key
        gcs_bucket      :       Name of a GCS Bucket to-be-generated
        google_project  :       Name of the google project this Cloud Function is in
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
    print(request_bytes)
    
    request_json = json.loads(request_bytes)
    print(request_json)    

    model_id = request_json['modelId']
    model_run_id = request_json['modelRunId']
    model_type = request_json['modelType']

    lb_client = get_lb_client()
    bucket = get_gcs_client().create_bucket(env_vars('gcs_bucket'), location = 'US-CENTRAL1')

    json_data = etl_job(lb_client, model_run_id, bucket)
    gcs_key = create_gcs_key(model_run_id)
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    
    print(f'ETL File: {etl_file}')
    
    post_bytes = json.dumps({str(model_run_id) : str(etl_file)}).encode('utf-8')
    train_url = env_vars("train_url")
    requests.post(train_url, data=post_bytes)
    
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
