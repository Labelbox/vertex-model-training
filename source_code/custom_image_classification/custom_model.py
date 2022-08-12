import tensorflow as tf
from tensorflow.python.client import device_lib
from labelbox import Client
from labelbox.data.serialization import LBV1Converter
from labelbox.data.annotation_types import Label, Radio
from labelbox.data.serialization import NDJsonConverter
from google.api_core.retry import Retry
from PIL.Image import Image, open as load_image, DecompressionBombError
import requests
from io import BytesIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Callable, Dict, Any, List
from tensorflow.keras.utils import to_categorical
import uuid

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--LB_API_KEY', dest='LB_API_KEY',
                    type=str,
                    help='Labelbox API KEY')
parser.add_argument('--model_id', dest='model_id',
                    type=str,
                    help='Labelbox model id to get ontology from.')
parser.add_argument('--model_run_id', dest='model_run_id',
                    type=str,
                    help='Labelbox model run id to get dataset from.')
parser.add_argument('--distribute', dest='distribute', type=str, default='single',
                    help='distributed training strategy')
args = parser.parse_args()

print(f"Labelbox model run id: {args.model_run_id}")
print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Single Machine, single compute device
if args.distribute == 'single':
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
# Single Machine, multiple compute device
elif args.distribute == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
# Multiple Machine, multiple compute device
elif args.distribute == 'multi':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Multi-worker configuration
print('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))

NUM_WORKERS = strategy.num_replicas_in_sync
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size.
MODEL_DIR = os.getenv("AIP_MODEL_DIR")

# Model config and Hyperparamters (will come from model run config in future)
BATCH_SIZE = 16
EPOCHS = 5


def build_model(num_classes):
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


class InvalidDataRowException(Exception):
    """Raised whenever the contents of a data row is either inaccessible or is too large"""
    pass


class InvalidLabelException(Exception):
    """ Exception for when the data is invalid for vertex."""
    pass


def process_label(label: Label, resize_w=256, resize_h=256) -> Dict[str, Any]:
    """ Function for converting a labelbox Label object into a vertex json label for single classification.
    Args:
        label: the label to convert
        downsample_factor: how much to scale the images by before running inference
    Returns:
        Dict representing a vertex label
    Nested Functions:
        get_image_bytes()     
        upload_image_to_gcs()
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
        classification = 'no_label'
    else:
        classification = classifications[0]
    split = label.extra.get("Data Split")
    data_row_id = label.data.uid
    return image_np_array, classification, split, data_row_id


def get_image_np_array(image_url: str, resize_w=256, resize_h=256) -> Optional[Tuple[Image, Tuple[int, int]]]:
    """ Fetch image bytes from a url and optionally resize the image.
    Args:
        image_url: A url that references an image

    Returns:
        The resized PIL Image
    Nested Functions:
        _download_image()
        image_to_bytes()
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


def process_labels_in_threadpool(process_fn: Callable[..., Dict[str, Any]], labels: List[Label], *args, max_workers=8) -> List[Dict[str, Any]]:
    """ Function for running etl processing in parallel
    Args:
        process_fn: Function to execute in parallel. Should accept Label as the first param and then any optional number of args.
        labels: List of labels to process
        args: Args that are passed through to the process_fn
        max_workers: How many threads should be used
    Returns:
        A list of results from the process_fn       
    """
    print('Processing Labels')
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    label2number = {
        "waffles": 0,
        "pancakes": 1,
        "omelette": 2,
    }
    data_row_ids_per_split = {
        "training": [],
        "test": [],
        "validation": []
    }
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        training_data_futures = (exc.submit(
            process_fn, label, *args) for label in labels)
        filter_count = {'labels': 0, 'data_rows': 0}
        for future in as_completed(training_data_futures):
            try:
                image_arr, classification_str, split, data_row_id = future.result()
                if split == "training":
                    x_train.append(image_arr)
                    y_train.append(label2number[classification_str])
                elif split == "test":
                    x_test.append(image_arr)
                    y_test.append(label2number[classification_str])
                elif split == "validation":
                    x_val.append(image_arr)
                    y_val.append(label2number[classification_str])
                # keep data row ids so that we can upload predictions back later
                data_row_ids_per_split[split].append(data_row_id)
            except InvalidDataRowException:
                filter_count['data_rows'] += 1
            except InvalidLabelException:
                filter_count['labels'] += 1
    x_train = np.stack(x_train, axis=0)
    x_test = np.stack(x_test, axis=0)
    x_val = np.stack(x_val, axis=0)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    print('Label Processing Complete')
    return x_train, y_train, x_test, y_test, x_val, y_val, data_row_ids_per_split


def get_options(model_id, lb_client):
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
    classifications = ontology.classifications()
    options = {}
    for classification in classifications:
        options.update({
            f"{option.value}": {
                "feature_schema_id": option.feature_schema_id,
                "parent_feature_schema_id": classification.feature_schema_id,
                "type": classification.class_type.value
            } for option in classification.options
        })
    return options


def build_radio_ndjson(confidences, options, data_row_id):
    """
    Args:
    Returns:
    """
    number2label = {
        0: "waffles",
        1: "pancakes",
        2: "omelette",
    }
    argmax = np.argmax(confidences)
    predicted = number2label[argmax]

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


def get_predictions(lb_model_id, model, data_by_split):
    options = get_options(lb_model_id, client)
    predictions = []
    for split, data in data_by_split.items():
      results = []
      for i in range(0, data.shape[0], BATCH_SIZE):
        if (i+1)*BATCH_SIZE > data.shape[0]:
          results.append(model.predict_on_batch(data[i*BATCH_SIZE:]))
        else:
          results.append(model.predict_on_batch(data[i*BATCH_SIZE: (i+1)*BATCH_SIZE]))
      results = np.concatenate(results, axis=0)
      for i, res in enumerate(results):
        data_row_id = data_row_ids_per_split[split][i]
        predictions.append(build_radio_ndjson(res, options, data_row_id))
    return predictions

if __name__ == "__main__":

    with strategy.scope():
        # Creation of dataset, and model building/compiling need to be within
        # `strategy.scope()`.
        model = build_model(num_classes=3)
        print("Model built")
        print(model.summary)

    client = Client(args.LB_API_KEY, enable_experimental=True)
    model_run = client.get_model_run(args.model_run_id)
    print("Exporting data")
    model_run.update_status("EXPORTING_DATA")
    labels_generator = get_labels_for_model_run(
        client, model_run_id=args.model_run_id, media_type="image", strip_subclasses=True)

    print("Preparing data")
    model_run.update_status("PREPARING_DATA")
    x_train, y_train, x_test, y_test, x_val, y_val, data_row_ids_per_split = process_labels_in_threadpool(
        process_label, labels_generator)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(BATCH_SIZE)

    print("training model")
    model_run.update_status("TRAINING_MODEL")
    history = model.fit(train_data, epochs=EPOCHS)

    print("inference on trained model")
    data_by_split = {"training": x_train, "validation": x_val, "test": x_test}
    predictions = get_predictions(args.model_id, model, data_by_split)

    print("uploading predictions to model run")
    task = model_run.add_predictions("upload predictions", predictions)
    print("prediction task errors:", task.errors)

    print("done")
    model_run.update_status("COMPLETE")
    model.save(MODEL_DIR)
