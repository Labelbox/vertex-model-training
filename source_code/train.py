https://cloud.google.com/vertex-ai/docs/training/automl-api#aiplatform_create_training_pipeline_image_classification_sample-python

def create_autoML_training_job(name: str, vertex_dataset_id):
  
  job = aiplatform.AutoMLImageTrainingJob(
    display_name = name,
    prediction = "classification",
    multi_label = False,
    model_type = "MOBILE_TF_VERSATILE_1"
  )
  
  model = job.run(
    dataset = aiplatform.ImageDataset(vertex_dataset_id),
    model_display_name = name,
    budget_milli_node_hours = 20000,
    training_filter_split = "labels.aiplatform.googleapis.com/ml_use=training",
    validation_filter_split = "labels.aiplatform.googleapis.com/ml_use=validation",
    test_filter_split = "labels.aiplatform.googleapis.com/ml_use=test"
  )
  
  model.wait()
