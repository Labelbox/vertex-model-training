def models(request):
    """To-be-used in a Google Cloud Function.
    Args:
        model_options           :           A list of model names you want to appear in the Labelbox UI
    """

    model_options = [ ## Input list of model options here
        "image_classification_custom_model"
    ]

    models_dict = {}

    for model in model_options:
        return_value.update({model : []})

    return models_dict