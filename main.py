def inference_function(request):
    """ Generates and uploads predictions to a given model run
    Args (passed from the monitor_function):
        lb_api_key          :           Labelbox API Key
        etl_file            :           URL to the ETL'ed data row / ground truth data from a Labelbox Model Run
        model_name          :           Display name used for the Vertex AI Model
        lb_model_run_id     :           Labelbox Model Run ID - used as the batch predction job name
    """
    import json
    import uuid
    from labelbox import Client, ModelRun
    from labelbox.data.serialization import NDJsonConverter
    from google.cloud import aiplatform    
    from source_code.config import env_vars

    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    lb_api_key = request_json['lb_api_key']    
    etl_file = request_json['etl_file'] 
    model_name = request_json['model_name'] 
    lb_model_run_id = request_json['lb_model_run_id'] 
    model_type = request_json['model_type']
    
    lb_client = Client(lb_api_key)
    model_run = lb_client._get_single(ModelRun, lb_model_run_id)
    
    if model_type == "autoML_image_classification":
        from source_code.autoML_image_classification.inference import batch_predict, get_options, process_predictions, export_model_run_labels, compute_metrics
    elif model_type == "custom_image_classification":
        from source_code.custom_image_classification.inference import batch_predict, get_options, process_predictions, export_model_run_labels, compute_metrics
    
    model = aiplatform.Model.list(filter=f'display_name={model_name}')[0]
    prediction_job = batch_predict(etl_file, model, lb_model_run_id, "radio")
    print('Predictions generated. Converting predictions into Labelbox format.')
    options = get_options(model_run.model_id, lb_client)
    annotation_data = process_predictions(prediction_job, options)
    predictions = list(NDJsonConverter.deserialize(annotation_data))
    print('Predictions reformatted. Exporting ground truth labels from model run.')
    labels = export_model_run_labels(lb_client, lb_model_run_id, 'image')
    print('Computing metrics.')    
    predictions_with_metrics = compute_metrics(labels, predictions, options)
    content = list(NDJsonConverter.serialize(predictions_with_metrics))
    print(content)
    print('Metrics computed. Uploading predictions and metrics to model run.')   
    upload_task = model_run.add_predictions(f'diagnostics-import-{uuid.uuid4()}', content)
    print(upload_task.statuses)
    print('Inference job complete.')
    
    return "Inference Job"

def monitor_function(request):
    """ Periodically checks a training job to see if it's completed, canceled, paused or failing
    Args (passed from the train_function):
        model_name          :           Display name used for the Vertex AI Model
        inference_url       :           URL that will trigger the inference function
        monitor_url         :           URL that will trigger the monitor function to run again
    Returns:
        Will either send the model training pipeline to inference or terminate the model training pipeline
    """    
    import requests
    import json
    import time
    from google.cloud import aiplatform
    from source_code.config import env_vars 
    
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)    
    model_name = request_json["model_name"]
    inference_url = request_json["inference_url"]
    monitor_url = request_json["monitor_url"]
    model_type = request_json['model_type']
    
    time.sleep(300)

    if model_type == "autoML_image_classification":
        training_job = aiplatform.AutoMLImageTrainingJob.list(filter=f'display_name={model_name}')[0]
    elif model_type == "custom_image_classification":
        training_job = aiplatform.CustomTrainingJob.list(filter=f'display_name={model_name}')[0]
    
    job_state = str(training_job.state)
    
    completed_states = [
        "PipelineState.PIPELINE_STATE_FAILED",
        "PipelineState.PIPELINE_STATE_CANCELLED",
        "PipelineState.PIPELINE_STATE_PAUSED",
        "PipelineState.PIPELINE_STATE_CANCELLING"
    ]
    
    print(f'Current Job State: {job_state}')
    
    if job_state == "PipelineState.PIPELINE_STATE_SUCCEEDED":
        print('Training compete, sent to inference.')
        requests.post(inference_url, data=request_bytes)
    elif job_state in completed_states:
        print("Training failed, terminating deployment.")
    else:
        print('Training incomplete, will check again in 5 minutes.')
        requests.post(monitor_url, data=request_bytes)

    return "Monitor Job"

def train_function(request):
    """ Initiates the training job in Vertex
    Args (passed from the etl_function):
        etl_file            :           URL to the ETL'ed data row / ground truth data from a Labelbox Model Run    
        lb_model_run_id     :           Labelbox Model Run ID - used as the training job name        
        model_name          :           Display name used for the Vertex AI Model
        monitor_url         :           URL that will trigger the monitor function 
    Returns:
        Creates a vertex dataset and launches a Vertex training job, triggers the monitor function
    """   
    import json
    import requests
    from source_code.config import env_vars    
    
    
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    etl_file = request_json['etl_file']
    lb_model_run_id = request_json['lb_model_run_id']
    model_name = request_json['model_name']
    monitor_url = request_json['monitor_url']
    model_type = request_json['model_type']

    if model_type == "autoML_image_classification":
        from source_code.autoML_image_classification.train import create_vertex_dataset, create_training_job
    elif model_type == "custom_image_classification":
        from source_code.custom_image_classification.train import create_vertex_dataset, create_training_job 
    
    vertex_dataset = create_vertex_dataset(lb_model_run_id, etl_file)
    vertex_model, vertex_model_id = create_training_job(model_name, vertex_dataset, lb_model_run_id)
    
    print('Training launched, sent to monitor function.')                                                              
    print(f"Job Name: {lb_model_run_id}")
    print(f'Vertex Model ID: {vertex_model_id}')
    
    requests.post(monitor_url, data=request_bytes)
    
    return "Train Job"

def etl_function(request):
    """ Exports data rows and ground truth labels from a model run, generates an ETL file in a storage bucket and launches training
    Args:
        lb_api_key          :           Labelbox API Key    
        gcs_bucket          :           Creates a gcs bucket with this name. Ensure that this bucket doesn't exist yet, or ETL will fail
        google_project      :           Name of the google project where this Cloud Function is hosted
        model_name          :           Display name used for the Vertex AI Model
        train_url           :           URL that will trigger the training function
        monitor_url         :           URL that will trigger the monitor function      
        inference_url       :           URL that will trigger the inference function  
    Returns:
        Google Bucket with an ETL file representing the data rows and ground truth labels from the model run
        Dictionary that gets passed through the other functions
    """
    import json
    import requests    
    from labelbox import Client    
    from google.cloud import storage, aiplatform    
    from source_code.config import env_vars
    
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    lb_model_id = request_json['modelId']
    lb_model_run_id = request_json['modelRunId']
    model_type = request_json['modelType']
    
    lb_api_key = env_vars("lb_api_key")
    lb_client = get_lb_client(lb_api_key)
    google_project = env_vars("google_project")
    gcs_bucket = env_vars('gcs_bucket')
    bucket = get_gcs_client(google_project).create_bucket(gcs_bucket, location = 'US-CENTRAL1') 
    gcs_key = create_gcs_key(lb_model_run_id)   

    if model_type == "autoML_image_classification":
        from source_code.autoML_image_classification.etl import etl_job, upload_ndjson_data
    elif model_type == "custom_image_classification":
        from source_code.custom_image_classification.etl import etl_job, upload_ndjson_data
    
    print("Beginning ETL")
    
    json_data = etl_job(lb_client, lb_model_run_id, bucket)
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    
    print(f'ETL File: {etl_file}')
    
    post_dict = {
        "model_type" : model_type,
        "lb_model_id" : lb_model_id,
        "lb_model_run_id" : lb_model_run_id,
        "etl_file" : etl_file,
        "lb_api_key" : lb_api_key,
        "google_project" : env_vars("google_project"),
        "model_name" : env_vars("model_name"),
        "train_url" : env_vars('train_url'),
        "monitor_url" : env_vars('monitor_url'),
        "inference_url" : env_vars('inference_url')
    }
    
    post_bytes = json.dumps(post_dict).encode('utf-8')
    requests.post(post_dict['train_url'], data=post_bytes)
    
    print(f"ETL Complete. Training Job Initiated.")
    
    return "ETL Job"

def model_run(request):
    """ Reroutes the webhook trigger to the ETL function
    Args:
        etl_url             :           URL that will trigger the etl function
    """
    import requests
    from source_code.config import env_vars
    
    string = request.get_data()
    etl_url = env_vars("etl_url")
    
    requests.post(etl_url, data=string)

    return "Rerouting to ETL"

def models(request):
    """ Serves a list of model options in the Labelbox UI
    Args:
        model_options       :           A list of model names you want to appear in the Labelbox UI
    """
    model_options = [ ## Input list of model options here
        "autoML_image_classification",
        "custom_image_classification"
    ]

    models_dict = {}

    for model in model_options:
        models_dict.update({model : []})

    return models_dict
