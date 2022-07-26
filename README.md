# vertex-model-training

This repository is a one stop shop for integrating Labelbox with a model Training backend on Google VertexAI (AutoML).  

### Background
Labelbox is designed to integrate with any Model Training or Pipeline backend.  Whether or not your existing backend supports an Active Learning lifecycle, Labelbox integration can improve your training data MLDLC by adding the following features:

- Improved error analysis leveraging confidence threshold support (5x/10x improvement to error analysis)
- Pre integrated MAL
- Models training limits as high as 1M data rows 
- Aligning your labeling data curation with your data split management
- Model Runs provide a complete historical to the assets, annotations, test/train/validate splits for every model training run
- Integrated launch of Model Training from the Labelbox Interface. Labelbox becomes the IDE and data debugger for training data curation
- Data Selection, whether to improve active learning cycle time, or prioritize production bugfix
- Integrates with MLDLC and CI/CD pipelines
- Data discovery with Catalog: Load all historical Model inference to Labelbox. The seamless integration , so that they get teh biggest model improvement for their $/effort/time

### Data Modality
Image Object Detection

### Dependencies: 
- [Labelbox SDK/Webhooks](https://docs.labelbox.com/docs/webhooks)  
- [VertexAI SDK](https://cloud.google.com/python/docs/reference/aiplatform/latest)
- [Google Cloud Functions](https://cloud.google.com/functions)
- Future Consideration: Vertex Pipelines

### How it works
Once the integration is set up, a model training sequence works as follows: 

1)	Initiate Model Training: User clicks "Train Model" from Model Runs Page
2)	Webhook Action Fired: Labelbox Model Run webhook is triggered.  The webhook calls Google Cloud Function.  Labelbox provides a sample Google Cloud Function (WIP) to give customers a fast "Hello World"
3)	Webhook ETL: The webhook callback calls Labelbox SDK to export data from the model run.  The data is then translated from Labelbox Format to VertexAI format and loadedd to Vertex as a Vertex AI Dataset. 
4)	Webhook Launch Model Training: VertexAI API is called to initiate the Model Training Job
5)	Webhook Polling Model Training Job: Webhook checks periodically for Model Training completion
6)	Webhook Inference on Test/Train splits: When model training is done, the trained model is invoked to run inference on the test/validate data splits
7)	Webhook Load results to Labelbox: Webhook handles calls Lablebox SDK to load test/validate inference and diagnostics enabling Labelbox's detailed visual model run analaysis

### How to set up in your own Labelbox / GCP envirionment
1) Set up (or select) a google project in GCS to host your Cloud Functions, take note of the google project name
2) Create a Labelbox API key
4) Create a 1st-Gen cloud function named 'monitor_function'
3) Create 2nd-Gen cloud functions named `etl_function`, `train_function`, and `inference_function`
4) Within each cloud function, take note of the URL on the `Trigger` tab for each cloud function
5) Once all that data is noted, run the following in your command-line-interface (for Macs, Terminal as an example)

`cd example_directory` (for example, Downloads)

`git clone https://github.com/Labelbox/vertex-model-training.git`

`cd vertex-model-training`

`gcloud functions deploy models --entry-point models --runtime python37 --trigger-http --allow-unauthenticated --timeout=540`

`gcloud functions deploy model_run --entry-point model_run --runtime python37 --trigger-http --allow-unauthenticated --timeout=540 --set-env-vars=etl_url=ETL_FUNCTION_URL` (insert your value for ETL_FUNCTION_URL)

`gcloud beta functions deploy etl-function --gen2 --entry-point etl_function --runtime python38 --trigger-http --allow-unauthenticated --timeout=3600 --set-env-vars=lb_api_key=API_KEY,gcs_bucket=BUCKET_NAME,model_name=MODEL_NAME,google_project=GOOGLE_PROJECT,train_url=TRAIN_FUNCTION_URL,monitor_url=MONITOR_FUNCTION_URL,inference_url=INFERENCE_FUNCTION_URL` (insert your values for API_KEY, BUCKET_NAME, MODEL_NAME, GOOGLE_PROJECT, TRAIN_FUNCTION_URL, MONITOR_FUNCTION_URL, and INFERENCE_FUNCTION_URL)

`gcloud beta functions deploy train-function --gen2 --entry-point train_function --runtime python38 --trigger-http --allow-unauthenticated --timeout=3600`

`gcloud functions deploy monitor-function --entry-point monitor_function --runtime python37 --trigger-http --allow-unauthenticated --timeout=540`

`gcloud beta functions deploy inference-function --gen2 --entry-point inference_function --runtime python38 --trigger-http --allow-unauthenticated --timeout=3600 --memory=8192MB`
