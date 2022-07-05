def model_run(request):
    """
    Args:
        
    """

    from labelbox import Client
    import json
    
    string = request.get_data()
    request_data = json.loads(string)
    model_id = request_data['modelId']
    model_run_id = request_data['modelRunId']
    model_type = request_data['modelType']

    return