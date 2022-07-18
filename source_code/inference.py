# https://cloud.google.com/vertex-ai/docs/predictions/batch-predictions#image_1

from labelbox import Client
from source_code.inference import batch_predict, get_options, process_predictions, export_model_run_labels, compute_metrics

def inference_function(request):


    lb_client = Client(request_json['lb_api_key'])
    prediction_job = batch_predict(etl_file, model, job_name, model_type)
    model_run = lb_client._get_single(request_json['model_run_id'])
    options = get_options(model_run.model_id)
    annotation_data = process_predictions(prediction_job, options)
    predictions = list(NDJsonConverter.deserialize(annotation_data))
    labels = export_model_run_labels(lb_client, model_run_id, 'image')
    metrics = compute_metrics(labels, predictions, options)
    upload_task = model_run.add_predictions(
      f'diagnostics-import-{uuid.uuid4()}',
       NDJsonConverter.serialize(predictions))
    upload_task.wait_until_done()
    
    return "Inference Job"
