import requests
import uuid
import ndjson
import time
from labelbox.data.serialization import LBV1Converter, NDJsonConverter
from labelbox.data.metrics.group import get_label_pairs
from labelbox.data.metrics import feature_miou_metric, feature_confusion_matrix_metric
from google.cloud import storage, aiplatform

def compute_metrics(labels, predictions, options):
    """ Computes metrics and adds metric values to predictions to-be-uploaded to Labelbox
    Args:
        labels      :       List of NDJSON ground truth labels from a model run
        predictions :       List of NDJSON prediction labels from a ETL'ed prediction job
        options     :       A dictionary where you can lookup the name of an option given the schemaId
    Returns:
        The same prediction list passed in with metrics attached, ready to-be-uploaded to a model run
    Nested Function:
        add_name_to_annotation()
    """
    predictions_with_metrics = []
    pairs = get_label_pairs(labels, predictions, filter_mismatch=True)
    for (ground_truth, prediction) in pairs.values():
        metrics = []
        for annotation in prediction.annotations:
            add_name_to_annotation(annotation, options)
        for annotation in ground_truth.annotations:
            add_name_to_annotation(annotation, options)
        metrics.extend(feature_confusion_matrix_metric(ground_truth.annotations, prediction.annotations))
        metrics.extend(feature_miou_metric(ground_truth.annotations, prediction.annotations))
        prediction.annotations.extend(metrics)
        predictions_with_metrics.append(prediction)
    return predictions_with_metrics

def add_name_to_annotation(annotation, options):
    """ Computes metrics and adds metric values to predictions to-be-uploaded to Labelbox
    Args:
        annotation      :       Annotation from a Labelbox NDJSON
        options         :       A dictionary where you can lookup the name of an option given the schemaId
    Returns:
        The same annotation with the name of the feature added in
    """    
    classification_name_lookup = {v['feature_schema_id']: k for k, v in options.items()}
    annotation.name = " "
    annotation.value.answer.name = classification_name_lookup[annotation.value.answer.feature_schema_id].replace(' ', '-')      

def export_model_run_labels(lb_client, model_run_id, media_type):
    """ Exports ground truth annotations from a model run
    Args:
        lb_client           :       Labelbox Client object
        model_run_id        :       Labelbox model run ID to pull data rows and ground truth labels from
        media_type          :       String that is either 'text' or 'image'
    Returns:
        NDJSON list of ground truth annotations from the model run
    """        
    query_str = """
        mutation exportModelRunAnnotationsPyApi($modelRunId: ID!) {
            exportModelRunAnnotations(data: {modelRunId: $modelRunId}) {
                downloadUrl createdAt status
            }
        }
        """
    url = lb_client.execute(query_str, {'modelRunId': model_run_id}, experimental=True)['exportModelRunAnnotations']['downloadUrl']
    counter = 1
    while url is None:
        counter += 1
        if counter > 10:
            raise Exception(f"Unsuccessfully got downloadUrl after {counter} attempts.")
        time.sleep(10)
        url = lb_client.execute(query_str, {'modelRunId': model_run_id}, experimental=True)['exportModelRunAnnotations']['downloadUrl']
    response = requests.get(url)
    response.raise_for_status()
    contents = ndjson.loads(response.content)
    for row in contents:
        row['media_type'] = media_type
    return LBV1Converter.deserialize(contents)

def process_predictions(batch_prediction_job, options):
    """
    Args:
    Returns:    
    Nested Functions:
        build_radio_ndjson()
    """
    annotation_data = []
    for batch in batch_prediction_job.iter_outputs():
        for prediction_data in ndjson.loads(batch.download_as_string()):
            if 'error' in prediction_data:
                continue
            prediction = prediction_data['prediction']
            # only way to get data row id is to lookup from the content uri
            data_row_id = prediction_data['instance']['content'].split("/")[-1].replace(".jpg", "")
            annotation_data.append(build_radio_ndjson(prediction, options, data_row_id))
    return annotation_data

def build_radio_ndjson(prediction, options, data_row_id):
    """
    Args:
    Returns:
    """    
    confidences = prediction['confidences']
    argmax = confidences.index(max(confidences))
    predicted = prediction['displayNames'][argmax]
    return {
        "uuid": str(uuid.uuid4()),
        "answer": {
            'schemaId': options[predicted]['feature_schema_id']
        },
        'dataRow': {
            "id": data_row_id
        },
        "schemaId": options[predicted]['parent_feature_schema_id']
    }

def get_options(model_id, lb_client):
    """
    Args:
    Returns:
    """
    ontology_id = lb_client.execute(
        """query modelOntologyPyApi($modelId: ID!){
            model(where: {id: $modelId}) {ontologyId}}
        """, {'modelId': model_id})['model']['ontologyId']
    ontology = lb_client.get_ontology(ontology_id)
    classifications = ontology.classifications()
    options = {}
    for classification in classifications:
        options.update({
            f"{classification.instructions}_{option.value}": {
                "feature_schema_id": option.feature_schema_id,
                "parent_feature_schema_id": classification.feature_schema_id,
                "type": classification.class_type.value
            } for option in classification.options
        })
    return options

def batch_predict(etl_file, model, job_name, model_type):
    """
    Args:
        etl_file        :       File generated from ETL function 
    Returns:
        
    Nested Functions:
      parse_url()
      build_inference_fule()
    """
    bucket_name, key = parse_uri(etl_file)
    source_uri = build_inference_file(bucket_name, key)
    nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    destination = f"gs://{bucket_name}/inference/{model_type}/{nowgmt}/"
    batch_prediction_job = model.batch_predict(
        job_display_name=job_name,
        instances_format='jsonl',
        machine_type='n1-standard-4',
        gcs_source=[source_uri],
        gcs_destination_prefix=destination,
        sync=False)
    batch_prediction_job.wait_for_resource_creation()
    while batch_prediction_job.state == aiplatform.compat.types.job_state.JobState.JOB_STATE_RUNNING:
        time.sleep(30)
    batch_prediction_job.wait()
    return batch_prediction_job

def build_inference_file(bucket_name : str, key: str) -> str:
    """ 
    Args:
        bucket_name         :        GCS bucket where the predictions will be saved
        key                 :        GCS key
    Returns:
        Inference file URL
    """        
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob(key)
    contents = ndjson.loads(blob.download_as_string())
    prediction_inputs = []
    for line in contents:
        prediction_inputs.append({
            "content": line['imageGcsUri'],
            "mimeType": "text/plain",
        })
    nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    blob = bucket.blob(f"inference_file/bounding-box/{nowgmt}.jsonl")
    blob.upload_from_string(data=ndjson.dumps(prediction_inputs), content_type="application/jsonl")
    return f"gs://{bucket.name}/{blob.name}"

def parse_uri(etl_file):
    """ Given an etl_file URI will return the bucket name and gcs key
    Args:
        etl_file            :       String URL representing the 
    Returns:
        Google storage bucket name and gcs key
    """     
    parts = etl_file.replace("gs://", "").split("/")
    bucket_name, key = parts[0], "/".join(parts[1:])
    return bucket_name, key
