def inference_function(request):
    import json
    import uuid
    from labelbox import Client
    from labelbox.data.serialization import NDJsonConverter
    from source_code.config import env_vars
    from source_code.inference import batch_predict, get_options, process_predictions, export_model_run_labels, compute_metrics   
    from google.cloud import aiplatform
    from labelbox import ModelRun

#     request_bytes = request.get_data()
#     request_json = json.loads(request_bytes)
    
    lb_api_key = env_vars('lb_api_key')
    etl_file = env_vars('etl_file')
    model_name = env_vars('model_name')
    lb_model_run_id = env_vars('lb_model_run_id')
    lb_client = Client(lb_api_key)
    
    model = aiplatform.Model.list(filter=f'display_name={model_name}')[0]
#     prediction_job = batch_predict(etl_file, model, lb_model_run_id, "radio")
    prediction_job = aiplatform.jobs.BatchPredictionJob.list(filter=f'display_name={lb_model_run_id}')[0]
    print('Predictions generated. Converting predictions into Labelbox format.')
    model_run = lb_client._get_single(ModelRun, lb_model_run_id)
    options = get_options(model_run.model_id)
    annotation_data = process_predictions(prediction_job, options)
    predictions = list(NDJsonConverter.deserialize(annotation_data))
    print('Predictions reformatted. Exporting ground truth labels from model run.')
    labels = export_model_run_labels(lb_client, lb_model_run_id, 'image')
    print('Computing metrics.')    
    compute_metrics(labels, predictions, options)
    print('Metrics computed. Uploading predictions and metrics to model run.')   
    upload_task = model_run.add_predictions(f'diagnostics-import-{uuid.uuid4()}', NDJsonConverter.serialize(predictions))
    upload_task.wait_until_done()
    print('Inference job complete.')
    return "Inference Job"

def monitor_function(request):
    import json
    import time
    from google.cloud import aiplatform
    import requests
    from source_code.config import env_vars 
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    time.sleep(300)
    training_job = aiplatform.AutoMLImageTrainingJob.list(filter=f'display_name={request_json["model_name"]}')[0]
    job_state = str(training_job.state)
    
    completed_states = [
        "PipelineState.PIPELINE_STATE_FAILED",
        "PipelineState.PIPELINE_STATE_CANCELLED",
        "PipelineState.PIPELINE_STATE_PAUSED",
        "PipelineState.PIPELINE_STATE_CANCELLING"
    ]
    
    print(job_state)
    
    if job_state == "PipelineState.PIPELINE_STATE_SUCCEEDED":
        print('Training compete, send to inference')
        requests.post(env_vars("inference_url"), data=request_bytes)
    elif job_state in completed_states:
        print("Training failed, terminating deployment")
    else:
        print('Training incomplete, will check again in 5 minutes')
        requests.post(env_vars("monitor_url"), data=request_bytes)
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
    vertex_dataset = create_vertex_dataset(request_json['lb_model_run_id'], request_json['etl_file'])
    vertex_model, vertex_model_id = create_autoML_training_job(request_json['model_name'], vertex_dataset, request_json['lb_model_run_id'])
    print(f"Training Job Name: {request_json['model_name']}")
    print(f'Vertex Model ID: {vertex_model_id}')
    requests.post(env_vars('monitor_url'), data=request_bytes)
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
        "lb_model_id" : model_id,
        "lb_model_run_id" : model_run_id,
        "etl_file" : etl_file,
        "lb_api_key" : env_vars("api_key"),
        "google_project" : env_vars("google_project"),
        "model_name" : env_vars("model_name")
    }
    post_bytes = json.dumps(post_dict).encode('utf-8')
    requests.post(env_vars('train_url'), data=post_bytes)
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
