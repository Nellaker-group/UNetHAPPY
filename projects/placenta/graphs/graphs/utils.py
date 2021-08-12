def get_feature(feature, predictions, embeddings):
    if feature == "predictions":
        return predictions
    elif feature == "embeddings":
        return embeddings
    else:
        raise ValueError(f"No such feature {feature}")
