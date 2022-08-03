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
- Data discovery with Catalog: Load all historical Model inference to Labelbox. The seamless integration , so that they get the biggest model improvement for their $/effort/time

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
3) GCloud Prerequisite

- Install [GCloud Client commandline tool](https://cloud.google.com/sdk/docs/install)

- Enable **Service Usage API** and **Vertex AI API**
 on your GCloud project

- Authenticate GCloud in your terminal:

```
gcloud auth login
``` 

- [Recommended] Configure gcloud config:

```
gcloud config set project PROJECT_ID
gcloud config set functions/region REGION
```

4) Clone this repo

```
git clone https://github.com/Labelbox/vertex-model-training.git`
cd vertex-model-training
```

5) Set up the /models endpoint to query available models in your training environment

- See `main.py` for implementation of this endpoint

```
gcloud functions deploy models --entry-point models --runtime python37 --trigger-http --allow-unauthenticated --timeout=540
```





6) Set up the train, monitor, and inference functions

- It will ask to you enable `artifactregistry.googleapis.com` and `run.googleapis.com` API services. 
- Adjust the memory limits for your application for each function. 

```
gcloud beta functions deploy train-function --gen2 --entry-point train_function --runtime python38 --trigger-http --allow-unauthenticated --timeout=3600  --memory=2048MB

gcloud functions deploy monitor-function --entry-point monitor_function --runtime python37 --trigger-http --allow-unauthenticated --timeout=540

gcloud beta functions deploy inference-function --gen2 --entry-point inference_function --runtime python38 --trigger-http --allow-unauthenticated --timeout=3600 --memory=8192MB
```

7) Configure env variables and deploy ETL function

```
export TRAIN_FUNCTION_URL=$(gcloud functions describe train-function --gen2 | grep "uri: " | cut  -c 8-) 
export INFERENCE_FUNCTION_URL=$(gcloud functions describe inference-function --gen2 | grep "uri: " | cut  -c 8-)
export MONITOR_FUNCTION_URL=$(gcloud functions describe monitor-function | grep "url: " | cut  -c 8-)

export GCS_BUCKET="<GCS_BUCKET>"
export GCS_REGION="us-central1"

export MODEL_NAME="<MODEL_NAME>"

export GOOGLE_PROJECT="<GOOGLE_PROJECT"

export LB_API_KEY="<LB_API_KEY>"
```


```
gcloud beta functions deploy etl-function --gen2 --entry-point etl_function --runtime python38 --trigger-http --allow-unauthenticated --timeout=3600 --set-env-vars=lb_api_key=$LB_API_KEY,gcs_region=$GCS_REGION,gcs_bucket=$GCS_BUCKET,model_name=$MODEL_NAME,google_project=$GOOGLE_PROJECT,train_url=$TRAIN_FUNCTION_URL,monitor_url=$MONITOR_FUNCTION_URL,inference_url=$INFERENCE_FUNCTION_URL,  --memory=4096MB
```

8) Deploy the /model_run endpoint 
```
export ETL_FUNCTION_URL=$(gcloud functions describe etl-function --gen2 | grep "uri: " | cut  -c 8-) 

gcloud functions deploy model_run --entry-point model_run --runtime python37 --trigger-http --allow-unauthenticated --timeout=540 --set-env-vars=etl_url=$ETL_FUNCTION_URL
```

9) In the Labelbox Model run page, configure model training integration
- go to your `models` function in the Google Cloud project, note the URL on the trigger tab will have something along the lines of `https://us-central1-GOOGLE_PROJECT.cloudfunctions.net/models` -- take note of the URL except for the `/models` suffix
- In your Labelbox Model, add this URL in the URL field on the `Settings` > `Model Training` section (example is `https://us-central1-GOOGLE_PROJECT.cloudfunctions.net/` in this case). If using Cloud Functions in this approach, no secret key is necessary. 
-  Now you can execute model training from Labelbox. Note that this protocol creates a Google Bucket, so if you run it again, you'll have to rename your Google Vertex Model Name and Google Cloud Storage Bucket by rerunning the `gcloud` command line for the `etl-function`.

## Update 
If you just need to update environment variable, you can do that via the cloud function UI, or [commandline](https://cloud.google.com/sdk/gcloud/reference/functions/deploy#--update-env-vars) for each cloud function. 

If you changed the specifc python code in each of the entrypoint function, you should re-deploy the functions by running corresponding command again.

For instance, if you changed `etl_function` in main.py or any functions it calls, you should re-deploy it by calling this deploy command again. 
```
gcloud beta functions deploy etl-function --gen2 --entry-point etl_function --runtime python38 --trigger-http --allow-unauthenticated --timeout=3600 --set-env-vars=lb_api_key=$LB_API_KEY,gcs_region=$GCS_REGION,gcs_bucket=$GCS_BUCKET,model_name=$MODEL_NAME,google_project=$GOOGLE_PROJECT,train_url=$TRAIN_FUNCTION_URL,monitor_url=$MONITOR_FUNCTION_URL,inference_url=$INFERENCE_FUNCTION_URL,  --memory=4096MB

```




