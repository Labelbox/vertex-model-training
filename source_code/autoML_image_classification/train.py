from google.cloud import aiplatform

def create_training_job(model_name: str, vertex_dataset, model_run_id):
    """ Launches the vertex training job
    Args:
        model_name              :           Display name given to the training job and the Vertex ML model
        vertex_dataset          :           aiplatform.ImageDataset object (Vertex Dataset)
        model_run_id            :           Used to generate the vertex model ID, which is the "model_name-model_run_id"
    Returns:
        Vertex model object, vertex model ID string
    """  
    vertex_model_id = str(model_name) + "-" + str(model_run_id)
    job = aiplatform.AutoMLImageTrainingJob(
    display_name = model_name,
    prediction_type = "classification",
    multi_label = False,
    model_type = "MOBILE_TF_VERSATILE_1"
    )
    model = job.run(
    dataset = vertex_dataset,
    sync = False,
    model_id = vertex_model_id,
    model_display_name = model_name,
    budget_milli_node_hours = 20000,
    training_filter_split = "labels.aiplatform.googleapis.com/ml_use=training",
    validation_filter_split = "labels.aiplatform.googleapis.com/ml_use=validation",
    test_filter_split = "labels.aiplatform.googleapis.com/ml_use=test"
    )
    return model, vertex_model_id

def create_vertex_dataset(name: str, gcs_etl_file):
    """ Converts an GCS ETL into a Vertex Dataset
    Args:
        name                    :           Name of the dataset in Vertex
        gcs_source              :           ETL File
        import_schema_uri       :           Tells vertex the kind of prediciton task is taking place In this case, it's a single label classification
    Returns:
        aiplatform.ImageDataset object
    """
    print('Creating Vertex Dataset')
    vertex_dataset = aiplatform.ImageDataset.create(display_name=name, 
                                                    gcs_source=gcs_etl_file,
                                                    import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification)
    print(f'Created Vertex Dataset with name {name}')
    return vertex_dataset
