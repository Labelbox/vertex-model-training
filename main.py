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
    from source_code.etl import etl_job, upload_ndjson_data, create_vertex_dataset
    from source_code.train import create_autoML_training_job
    from labelbox import Client
    from google.cloud import storage
    from google.cloud import aiplatform
    
    string = request.get_data()
    print(string)
    request_data = json.loads(string)

    model_id = request_data['modelId']
    model_run_id = request_data['modelRunId']
    model_type = request_data['modelType']
    
    print(model_run_id)

    lb_client = get_lb_client()
    bucket = get_gcs_client().create_bucket(env_vars('gcs_bucket'), location = 'US-CENTRAL1')

    json_data = etl_job(lb_client, model_run_id, bucket)
    gcs_key = create_gcs_key(model_run_id)
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    vertex_dataset = create_vertex_dataset(model_run_id, etl_file)
    
    vertex_model, vertex_model_id = create_autoML_training_job("test_model", vertex_dataset, model_run_id)
    
    print(vertex_model_id)
    
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
