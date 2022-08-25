from labelbox import Client
from labelbox.data.serialization import LBV1Converter
from labelbox.data.annotation_types import Label, Radio
from labelbox.data.serialization import NDJsonConverter

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import to_categorical
from google.api_core.retry import Retry

from PIL.Image import Image, open as load_image, DecompressionBombError
import requests
from io import BytesIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Callable, Dict, Any, List

import uuid
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--LB_API_KEY', dest='LB_API_KEY', type=str, help='Labelbox API KEY')
parser.add_argument('--LB_MODEL_ID', dest='LB_MODEL_ID', type=str, help='Labelbox model id to get ontology from.')
parser.add_argument('--LB_MODEL_RUN_ID', dest='LB_MODEL_RUN_ID', type=str, help='Labelbox model run id to get dataset from.')
parser.add_argument('--EPOCHS', dest='EPOCHS', type=int, default=5, help='Number of epochs to train the model on.')
parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', type=int, default=16, help='Training batch size.')
parser.add_argument('--DISTRIBUTE', dest='DISTRIBUTE', type=str, default='single', help='Which distributed training strategy to use in container generated.')
parser.add_argument('--MODEL_SAVE_DIR', dest='MODEL_SAVE_DIR', type=str, help='Where to save the trained model')

args = parser.parse_args()

print(f"Labelbox model run id: {args.LB_MODEL_RUN_ID}")
print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

def specify_compute_strategy(distribute_value):
    """ Returns a tensorflow compute strategy
    Args:
        distribute_value (str)      :       Either "single", "mirror" or "multi"
    Returns
        tensorflow.distribute strategy
    """
    # Single Machine, single compute device
    if distribute_value == 'single':
        if tf.test.is_gpu_available():
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    # Single Machine, multiple compute device
    elif distribute_value == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    # Multiple Machine, multiple compute device
    elif distribute_value == 'multi':
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    return strategy

def build_model(num_classes):
    """ Fine tunes an InceptionV3 model on a given number of classes
    Args:
        num_classes     :       Number of classes in your new ontology
    Returns:
        tf.kkeras.Model object
    """
    # In this example, we will finetune a InceptionV3 with imagenet pre-trained weights
    # create the base pre-trained model
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # and a logistic layer -- map to number of classes
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # Train only the top layers (which were randomly initialized)
    # i.e. freeze all pretrained convolutional InceptionV3 layers, and only finetune on the last 3 layers we added.
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    )
    return model

def get_labels_for_model_run(client: Client, model_run_id: str, media_type: str, strip_subclasses: bool):
    """ Exports all labels from a model run
    Args:
        client          :       Labelbox client used for fetching labels
        model_run_id    :       model run to fetch labels for
        media_type      :       Should either be "image" or "text" string
    Returns:
        LabelGenerator with labels to-be-converted into vertex syntax    
    """
    print("Initiating Label Export")
    model_run = client.get_model_run(model_run_id)
    json_labels = model_run.export_labels(download=True)
    print("Label Export Complete")
    for row in json_labels:
        if media_type is not None:
            row['media_type'] = media_type
        if strip_subclasses:
            # Strip subclasses for tools
            for annotation in row['Label']['objects']:
                if 'classifications' in annotation:
                    del annotation['classifications']
            # Strip subclasses for classifications
            for annotation in row['Label']['classifications']:
                if 'answer' in annotation:
                    if 'classifications' in annotation['answer']:
                        del annotation['answer']['classifications']
                if 'answers' in annotation:
                    for answer in annotation['answers']:
                        if 'classifications' in answer:
                            del answer['classifications']
    return LBV1Converter.deserialize(json_labels)

class InvalidDataRowException(Exception):
    """Raised whenever the contents of a data row is either inaccessible or is too large"""
    pass


class InvalidLabelException(Exception):
    """ Exception for when the data is invalid for vertex."""
    pass


def process_label(label, resize_w=256, resize_h=256):
    """ Takes a label from label.generator() and returns input_value, label_class_name, training_split, data_row_id
    """
    classifications = []
    image_np_array, _ = get_image_np_array(label.data.url, resize_w, resize_h)
    for annotation in label.annotations:
        if isinstance(annotation.value, Radio):
            classifications.append(
                annotation.value.answer.name
            )
    if len(classifications) > 1:
        raise InvalidLabelException(
            "Skipping example. Must provide <= 1 classification per image.")
    elif len(classifications) == 0:
        label_name = 'no_label'
    else:
        label_name = classifications[0]
    split = label.extra.get("Data Split")
    data_row_id = label.data.uid
    
    return image_np_array, label_name, split, data_row_id


def get_image_np_array(image_url: str, resize_w=256, resize_h=256) -> Optional[Tuple[Image, Tuple[int, int]]]:
    """ Fetches the numpy array representation of an image URL
    Args:
        image_url       :       A url that references an image
        resize_w        :       Desired numpy array width for training
        resize_w        :       Desired numpy array height for training
    Returns:
        NumPy array represenetation of an image
    """
    try:
        with _download_image(image_url) as image:
            w, h = image.size
            with image.resize((int(resize_w), int(resize_h))) as resized_image:
                return np.array(resized_image), (w, h)
    except DecompressionBombError:
        raise InvalidDataRowException(f"Image too large : `{image_url}`.")
    except:
        raise InvalidDataRowException(
            f"Unable to fetch image : `{image_url}`.")


@Retry()
def _download_image(image_url: str) -> Image:
    """ Downloads as a PIL Image object
    Args:
        image_url       :       String of a URL to-be-downloaded as a PIL Image object
    Returns:
        Image as a PIL Image object
    """
    return load_image(BytesIO(requests.get(image_url).content))


def process_labels_in_threadpool(process_fn, labels, label_encoder, *args, max_workers=8):
    """ Function for running etl processing in parallel
    Args:
        process_fn          :       Function to create training data in parallel - first argument is the label
        labels              :       List of labels to process
        label_encoder       :       Dictionary where key=label_name, value=encoded model output value
        max_workers         :       How many threads should be used
    Returns:
        A list of results from the process_fn       
    """
    print('Processing Labels')
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
    data_row_id_input = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        training_data_futures = (exc.submit(
            process_fn, label, *args) for label in labels)
        filter_count = {'labels': 0, 'data_rows': 0}
        for future in as_completed(training_data_futures):
            try:
                image_arr, label_name, split, data_row_id = future.result()
                if split == "training":
                    x_train.append(image_arr)
                    y_train.append(label_encoder[label_name])
                elif split == "test":
                    x_test.append(image_arr)
                    y_test.append(label_encoder[label_name])
                elif split == "validation":
                    x_val.append(image_arr)
                    y_val.append(label_encoder[label_name])
                # Dictionary where {key=data_row_id : value=input data as numpy array}
                data_row_id_input[data_row_id]['input_data'] = image_arr
            except InvalidDataRowException:
                filter_count['data_rows'] += 1
            except InvalidLabelException:
                filter_count['labels'] += 1

    x_train, x_test = np.stack(x_train, axis=0), np.stack(x_test, axis=0)
    x_val, y_val = np.stack(x_val, axis=0), to_categorical(y_val)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    print('Label Processing Complete')

    return x_train, y_train, x_test, y_test, x_val, y_val, data_row_id_input

def map_model_ontology(model_id, lb_client):
    """ Creates a dictionary where key = classification_option : value = { feature_schema_id, parent_feature_schema_id, type } from a Labelbox Ontology
    Args:
        model_id        :       Labelbox Model ID to pull an ontology from
        lb_client       :       Labelbox Client object
    Returns:
        Reference dictionary for a given ontology
    """
    ontology_id = lb_client.execute(
        """query modelOntologyPyApi($modelId: ID!){
            model(where: {id: $modelId}) {ontologyId}}
        """, {'modelId': model_id})['model']['ontologyId']
    ontology = lb_client.get_ontology(ontology_id)
    feature_map = map_features(ontology.normalized)
    return feature_map

def map_features(ontology_normalized):
  """ Given an ontology, returns a dictionary where the key = featureSchemaId and value = [node_name, color, type, encoded_value] or a Pandas DataFrame
  Args:
    ontology_normalized   :   Queried from a project using project.ontology().normalized
    export_type           :   Either "index" or "dataframe"
  Returns:
    Dictionary where key = node_name and values = encoded_layer, parent_name, parent_feature_schema_id, feature_schema_id, type  
  """
  feature_map = {}
  tools = ontology_normalized["tools"]
  classifications = ontology_normalized["classifications"]
  if tools:
    feature_map, nested_layer = layer_iterator(feature_map, tools)
  if classifications:
    feature_map, nested_layer = layer_iterator(feature_map, classifications)
  return feature_map

def layer_iterator(feature_map, node_layer, parent_feature_schema_id=None, parent_name=None):
  """ Iterative function to create a lookup for each node in a node_layer
  Args:
    feature_map                     :       building dictionary where node_name is the key
    node_layer                      :       list of normalized ontology dictionaries
    parent_feature_schema_id        :       parent feature schema id
    parent_name                     :       parent node name
  Returns:
    Either recursively calls this function for a nested layer, or returns the updated feature_map
  """
  for count, node in enumerate(node_layer):
    node_feature_schema_id = node['featureSchemaId']
    if "tool" in node.keys(): # Catches Tools
      node_type = node["tool"]
      node_name = node["name"]
      next_layer = node["classifications"]
    elif "instructions" in node.keys(): # Catches Classifications
      node_type = node["type"]
      node_name = node["instructions"]
      next_layer = node["options"]
    else: # Catches options
      node_name = node["label"]
      if "options" in node.keys(): # Catches Options with Nested Classifications
        next_layer = node["options"]
        node_type = "branch_option"
      else: # Catches Options with no Nested Classificaitons
        next_layer = []
        node_type = "leaf_option"
    # Appends this list to our building DataFrame
    feature_map.update(
        {node_name : {
         "encoded_value" : count,
         "parent_name" : parent_name, 
         "parent_feature_schema_id" : parent_feature_schema_id, 
         "feature_schema_id" : node_feature_schema_id,
         "type" : node_type, 
        }
      }
    )
    if next_layer:
      feature_map, nested_layer = layer_iterator(feature_map, next_layer, node_feature_schema_id, node_name) 
  return feature_map, next_layer


def get_predictions(tf_model, data_row_id_input, lb_ontology_index, label_decoder, batch_size):
    """ Given a tensorflow model and input data by split, will create predictions and return Labelbox-ready list of predictions to upload
    Args:
        tf_model                :       tf.keras.Model object
        data_row_id_input       :       Dictionary where {key=data_row_id : value='input_data'}
        lb_ontology_index       :       Dictionary where {key=class_name : value={'feature_schema_id', 'parent_feature_schema_id'}}
        label_decoder           :       Dictionary where {key=encoded_value : value=class_name} 
        batch_size              :       Number of data rows to run predictions on in the same batch
    Returns:
        List of ndjsons to upload to labelbox
    """
    data_row_ids = []
    predictions = []
    for i in range(0, len(input_data), batch_size):
        batch_input_data = []
        if i+batch_size > len(shape):
            end = len(input_data)
        else:
            end = i + batch_size
        for key in sorted(input_data)[i:end]:
            data_row_ids.append(key)
            batch_input_data.append(input_data[key]['input_data'])
        prediction_values.append(tf_model.predict_on_batch(batch_input_data))
    for count, prediction_value in enumerate(prediction_values):
        data_row_id = data_row_ids[count]
        predictions.append(build_radio_ndjson(prediction_value, lb_ontology_index, data_row_id, label_decoder))
    return predictions

def build_radio_ndjson(confidences, feature_map, data_row_id, label_decoder):
    """
    Args:
        confidences     :       Array of 1 dimension that represents indexed confidence scores for the classifying layer
        feature_map     :       Dictionary where {key=class_name : value={'feature_schema_id', 'parent_feature_schema_id'}}
        data_row_id     :       Labelbox Data Row ID
        label_decoder       :       Dictionary where {key=encoded_value : value=class_name} 
    Returns:
        NDJSON format for a labelbox annotation upload
    """
    predicted_value = np.argmax(confidences)
    predicted_label = label_decoder[predicted_value]

    return {
        "uuid": str(uuid.uuid4()),
        "answer": {
            'schemaId': feature_map[predicted_label]['feature_schema_id']
        },
        'dataRow': {
            "id": data_row_id
        },
        "schemaId": feature_map[predicted_label]['parent_feature_schema_id']
    }

if __name__ == "__main__":

    print(f'Connecting to Labelbox...')
    lb_client = Client(args.LB_API_KEY, enable_experimental=True)
    lb_model_run = lb_client.get_model_run(args.LB_MODEL_RUN_ID)
    
    try:
        lb_ontology_index = map_model_ontology(args.LB_MODEL_ID, lb_client)
        label_encoder = {}
        for name in lb_ontology_index.keys():
            if lb_ontology_index[name]['parent_name'] == "Food": ## PARENT NAME THAT WE'RE RUNNING PREDICTIONS ON
                label_encoder[name] = lb_ontology_index[name]['encoded_value']
        label_decoder = {v: k for k, v in label_encoder.items()}         
        
        print("Successfully connected to Labelbox Model. Building custom model..")
        
        strategy = specify_compute_strategy(args.DISTRIBUTE)    
        print('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            tf_model = build_model(num_classes=len(label_encoder))
            print(tf_model.summary)      

        print("Custom model built. Exporting training data from Labelbox...")
        lb_model_run.update_status("EXPORTING_DATA")
        labels_generator = get_labels_for_model_run(lb_client, model_run_id=lb_model_run.uid, media_type="image", strip_subclasses=True)

        print("Export complete. Preparing data for training...")
        lb_model_run.update_status("PREPARING_DATA")
        x_train, y_train, x_test, y_test, x_val, y_val, data_row_id_input = process_labels_in_threadpool(process_label, labels_generator, label_encoder)
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.batch(args.BATCH_SIZE)

        print("Data prepared. Training model...")
        lb_model_run.update_status("TRAINING_MODEL")
        tf_history = tf_model.fit(train_data, epochs=args.EPOCHS)

        print("Model training complete. Creating predictions with trained model...")
        data_by_split = {"training": x_train, "validation": x_val, "test": x_test}
        predictions = get_predictions(args.LB_MODEL_ID, tf_model, data_row_id_input, lb_client, label_decoder, args.BATCH_SIZE)

        print(f"Uploading predictions to Laeblbox Model Run {lb_model_run.uid}")
        task = lb_model_run.add_predictions("upload predictions", predictions)
        print("prediction task errors:", task.errors)

        print("Done")
        lb_model_run.update_status("COMPLETE")
        tf_model.save(args.MODEL_SAVE_DIR)
        
    except Exception as e:
        lb_model_run.update_status("FAILED")
        print('Model Training Failed.')
        print(e)
