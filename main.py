def model_run(request):
    """
    """
    import json
    from source_code.config import env_vars, create_gcs_key, get_lb_client, get_gcs_client
    from source_code.etl import etl_job, upload_ndjson_data
    from labelbox import Client
    from google.cloud import storage
    
    string = request.get_data()
    request_data = json.loads(string)

    model_id = request_data['modelId']
    model_run_id = request_data['modelRunId']
    model_type = request_data['modelType']

    lb_client = get_lb_client()
    bucket = get_gcs_client().create_bucket(env_vars('gcs_bucket'), location = 'US-CENTRAL1')

    json_data = etl_job(lb_client, model_run_id, bucket)
    gcs_key = create_gcs_key(model_run_id)
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    
    print(f"ETL Complete. URI: {etl_file}")

    return "ETL Job"

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
        return_value.update({model : []})

    return models_dict