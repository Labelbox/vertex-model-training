from google.cloud import aiplatform

def create_autoML_training_job(name: str, vertex_dataset, model_run_id):
  
  vertex_model_id = str("model-") +cr str(model_run_id)
  
  job = aiplatform.AutoMLImageTrainingJob(
    display_name = name,
    prediction = "classification",
    multi_label = False,
    model_type = "MOBILE_TF_VERSATILE_1"
  )
  
  model = job.run(
    dataset = vertex_dataset,
    sync = False,
    model_id = vertex_model_id,
    model_display_name = name,
    budget_milli_node_hours = 20000,
    training_filter_split = "labels.aiplatform.googleapis.com/ml_use=training",
    validation_filter_split = "labels.aiplatform.googleapis.com/ml_use=validation",
    test_filter_split = "labels.aiplatform.googleapis.com/ml_use=test"
  )
  
  return model, vertex_model_id
