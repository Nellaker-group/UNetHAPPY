from happy.models import inceptionresnetv2, resnet


def build_cell_classifer(model_name, out_features):
    if model_name == "inceptionresnetv2":
        model = inceptionresnetv2.build_inceptionresnetv2(out_features=out_features)
    elif model_name == "resnet-50":
        model = resnet.build_resnet(out_features=out_features, depth=50)
    else:
        raise ValueError(f"model type {model_name} not supported for cell classifer")
    return model
