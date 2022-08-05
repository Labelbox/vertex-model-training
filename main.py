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
    from labelbox import Client
    from labelbox.data.serialization import NDJsonConverter
    from google.cloud import aiplatform    
    from source_code.config import get_lb_client

    # Receive data from trigger
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    
    # Parse data from trigger    
    lb_api_key = request_json['lb_api_key']    
    etl_file = request_json['etl_file'] 
    model_name = request_json['model_name'] 
    lb_model_run_id = request_json['lb_model_run_id'] 
    model_type = request_json['model_type']
    
    lb_client = get_lb_client(lb_api_key)
    model_run = lb_client.get_model_run(lb_model_run_id)    
    
    try:
        # Select model type       
        if model_type == "autoML_image_classification":
            from source_code.autoML_image_classification.inference import batch_predict, get_options, process_predictions, export_model_run_labels, compute_metrics
        elif model_type == "custom_image_classification":
            from source_code.custom_image_classification.inference import batch_predict, get_options, process_predictions, export_model_run_labels, compute_metrics 
        # Code execution    
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
        print('Metrics computed. Uploading predictions and metrics to model run.')   
        upload_task = model_run.add_predictions(f'diagnostics-import-{uuid.uuid4()}', content)
        print(upload_task.statuses)
        model_run.update_status("COMPLETE")  
        print('Inference job complete.')
    except:
        model_run.update_status("FAILED") 
    
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
    from source_code.config import get_lb_client
    
    # Receive data from trigger    
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)    
    
    # Parse data from trigger    
    lb_api_key = request_json['lb_api_key']    
    model_name = request_json["model_name"]
    lb_model_run_id = request_json['lb_model_run_id']    
    inference_url = request_json["inference_url"]
    monitor_url = request_json["monitor_url"]
    model_type = request_json['model_type']
    
    # Configure environment
    lb_client = get_lb_client(lb_api_key)  
    model_run = lb_client.get_model_run(lb_model_run_id)

    try:
        # Select model type    
        if model_type == "autoML_image_classification":
            training_job = aiplatform.AutoMLImageTrainingJob.list(filter=f'display_name={model_name}')[0]
        elif model_type == "custom_image_classification":
            training_job = aiplatform.CustomTrainingJob.list(filter=f'display_name={model_name}')[0]
        # Code execution    
        time.sleep(300)
        job_state = str(training_job.state)
        completed_states = [
            "PipelineState.PIPELINE_STATE_FAILED",
            "PipelineState.PIPELINE_STATE_CANCELLED",
            "PipelineState.PIPELINE_STATE_PAUSED",
            "PipelineState.PIPELINE_STATE_CANCELLING"
        ]
        print(f'Current Job State: {job_state}')    
        # Trigger model training monitor function or model training inference function  
        if job_state == "PipelineState.PIPELINE_STATE_SUCCEEDED":
            print('Training compete, sent to inference.')
            requests.post(inference_url, data=request_bytes)
        elif job_state in completed_states:
            print("Training failed, terminating deployment.")
            model_run.update_status("FAILED")
        else:
            print('Training incomplete, will check again in 5 minutes.')
            requests.post(monitor_url, data=request_bytes)
    except:
        model_run.update_status("FAILED")            

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
    from source_code.config import get_lb_client    
    
    # Receive data from trigger
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    
    # Parse data from trigger
    etl_file = request_json['etl_file']
    lb_api_key = request_json['lb_api_key']
    lb_model_run_id = request_json['lb_model_run_id']
    model_name = request_json['model_name']
    monitor_url = request_json['monitor_url']
    model_type = request_json['model_type']
    
    # Configure environment
    lb_client = get_lb_client(lb_api_key)
    model_run = lb_client.get_model_run(lb_model_run_id)    

    try:
        # Select model type
        if model_type == "autoML_image_classification":
            from source_code.autoML_image_classification.train import create_vertex_dataset, create_training_job
        elif model_type == "custom_image_classification":
            from source_code.custom_image_classification.train import create_vertex_dataset, create_training_job 
        # Code execution
        vertex_dataset = create_vertex_dataset(lb_model_run_id, etl_file)
        vertex_model, vertex_model_id = create_training_job(model_name, vertex_dataset, lb_model_run_id)
        model_run.update_status("TRAINING_MODEL")          
        print('Training launched, sent to monitor function.')                                                              
        print(f"Job Name: {lb_model_run_id}")
        print(f'Vertex Model ID: {vertex_model_id}')
        # Trigger model training monitor function
        requests.post(monitor_url, data=request_bytes)
    except:
        model_run.update_status("FAILED")
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
    from source_code.config import get_lb_client, get_gcs_client, create_gcs_key
    
    # Receive data from trigger 
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    
    # Read environment variables   
    lb_api_key = request_json["lb_api_key"]
    google_project = request_json["google_project"]
    gcs_bucket = request_json['gcs_bucket']
    train_url = request_json['train_url']
    gcs_region = request_json['gcs_region']
    lb_model_run_id = request_json['lb_model_run_id']
    
    # Configure environment
    lb_client = get_lb_client(lb_api_key)
    gcs_client = get_gcs_client(google_project)   
    model_run = lb_client.get_model_run(lb_model_run_id)
    
    try:
        # Select model type
        if model_type == "autoML_image_classification":
            from source_code.autoML_image_classification.etl import etl_job, upload_ndjson_data
        elif model_type == "custom_image_classification":
            from source_code.custom_image_classification.etl import etl_job, upload_ndjson_data   
        # Code execution      
        print("Beginning ETL")     
        model_run.update_status("PREPARING_DATA")   
        gcs_key = create_gcs_key(lb_model_run_id)
        try:
            bucket = gcs_client.get_bucket(gcs_bucket)
        except: 
            print(f"Bucket does not exsit, will create one with name {gcs_bucket}")
            bucket = gcs_client.create_bucket(gcs_bucket, location=gcs_region)
        json_data = etl_job(lb_client, lb_model_run_id, bucket)
        etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
        print(f'ETL File: {etl_file}')
        # Trigger model training function  
        request_json.update({"etl_file" : etl_file})
        post_bytes = json.dumps(request_json).encode('utf-8')
        requests.post(train_url, data=post_bytes)
        print(f"ETL Complete. Training Job Initiated.")
    except:
        print("ETL Function Failed. Check your configuration and try again.")
        model_run.update_status("FAILED")
    
    return "ETL Job"

def model_run(request):
    """ Reroutes the webhook trigger to the ETL function
    Args:
        etl_url             :           URL that will trigger the etl function
    """
    import requests
    import json
    from source_code.config import env_vars, get_lb_client

    # Receive data from trigger   
    request_bytes = request.get_data()
    request_json = json.loads(request_bytes)
    
    # Parse data from trigger    
    lb_model_id = request_json['modelId']
    lb_model_run_id = request_json['modelRunId']
    model_type = request_json['modelType']    
    
#     try:
    # Structure config variable json
    post_dict = {
        "model_type" : model_type,
        "lb_model_id" : lb_model_id,
        "lb_model_run_id" : lb_model_run_id,
        "gcs_bucket" : env_vars("gcs_bucket"),
        "gcs_region" : env_vars("gcs_region"),
        "lb_api_key" : env_vars("lb_api_key"),
        "google_project" : env_vars("google_project"),
        "model_name" : env_vars("model_name"),
        "train_url" : env_vars("train_url"),
        "monitor_url" : env_vars("monitor_url"),
        "inference_url" : env_vars("inference_url")
    }
    post_bytes = json.dumps(post_dict).encode('utf-8')
    # Update model run status
    lb_client = get_lb_client(post_dict["lb_api_key"])
    model_run = lb_client.get_model_run(lb_model_run_id)
    model_run.update_status("EXPORTING_DATA")
    # Send data to ETL Function
    requests.post(env_vars("etl_url"), data=post_bytes)
#     except:
#         print("Model Run Function Failed. Check your Environment Variables and try again.")
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
