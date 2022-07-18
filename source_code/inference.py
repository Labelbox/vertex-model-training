from labelbox.data.serialization import LBV1Converter, NDJsonConverter
from labelbox.data.metrics.group import get_label_pairs
from labelbox.data.metrics import feature_confusion_matrix_metric
import requests
import uuid
import ndjson
import time

def compute_metrics(self, labels, predictions, options):
    """
    Nested Function:
      add_name_to_annotation
    """
    pairs = get_label_pairs(labels, predictions, filter_mismatch=True)
    for (ground_truth, prediction) in pairs.values():
        for annotation in prediction.annotations:
            add_name_to_annotation(annotation, options)
        for annotation in ground_truth.annotations:
            add_name_to_annotation(annotation, options)
        prediction.annotations.extend(feature_confusion_matrix_metric(ground_truth.annotations, prediction.annotations))

def add_name_to_annotation(annotation, options):
    tool_name_lookup = {v['feature_schema_id']: k for k, v in options.items()}
    annotation.name = " "
    annotation.value.answer.name = tool_name_lookup[annotation.value.answer.feature_schema_id].replace(' ', '-')      

def export_model_run_labels(lb_client, model_run_id, media_type: Union[Literal['image'], Literal['text']]):
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
  Nested Functions:
    build_radio_ndjson
  """
    annotation_data = []
    for batch in batch_prediction_job.iter_outputs():
        for prediction_data in ndjson.loads(batch.download_as_string()):
            if 'error' in prediction_data:
                continue
            prediction = prediction_data['prediction']
            # only way to get data row id is to lookup from the content uri
            data_row_id = prediction_data['instance']['content'].split(
                "/")[-1].replace(".txt" if self.media_type == "text" else ".jpg", "")
            annotation_data.append(
                self.build_radio_ndjson(prediction, options, data_row_id))
    return annotation_data

def build_radio_ndjson(prediction, options, data_row_id):
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

def get_options(model_id):
    ontology_id = self.lb_client.execute(
        """query modelOntologyPyApi($modelId: ID!){
            model(where: {id: $modelId}) {ontologyId}}
        """, {'modelId': model_id})['model']['ontologyId']
    ontology = self.lb_client.get_ontology(ontology_id)
    classifications = ontology.classifications()
    options = {}
    for classification in classifications:
        options.update({
            f"{tool.instructions}_{option.value}": {
                "feature_schema_id": option.feature_schema_id,
                "parent_feature_schema_id": classification.feature_schema_id,
                "type": classification.class_type.value
            } for option in tool.options
        })
    return options

def batch_predict(etl_file, model, job_name, model_type):
    """
    Nested Functions:
      parse_url
      build_inference_fule
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

def parse_uri(self, etl_file):
    parts = etl_file.replace("gs://", "").split("/")
    bucket_name, key = parts[0], "/".join(parts[1:])
    return bucket_name, key
