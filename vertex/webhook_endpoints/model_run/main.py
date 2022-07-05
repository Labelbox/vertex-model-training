def model_run(request):
    """
    """
    import json
    from source_code.config import env_vars, create_gcs_key, get_lb_client, get_gcs_client
    from source_code.etl import etl_job, upload_ndjson_data
    
    string = request.get_data()
    request_data = json.loads(string)

    model_id = request_data['modelId']
    model_run_id = request_data['modelRunId']
    model_type = request_data['modelType']

    api_key = env_vars('api_key')
    gcs_bucket = env_vars('gcs_bucket')

    lb_client = get_lb_client(api_key)
    bucket = get_gcs_client().bucket(gcs_bucket)

    json_data = etl_job(lb_client, model_run_id, bucket)
    gcs_key = create_gcs_key(model_run_id)
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    
    logger.info("ETL Complete. URI: %s", f"{etl_file}")

    return "Training Job"
