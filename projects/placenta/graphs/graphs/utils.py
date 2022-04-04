import torch

def get_feature(feature, predictions, embeddings):
    if feature == "predictions":
        return predictions
    elif feature == "embeddings":
        return embeddings
    else:
        raise ValueError(f"No such feature {feature}")

def send_graph_to_device(data, device, tissue_class=None):
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    if not tissue_class is None:
        tissue_class = torch.Tensor(tissue_class).type(torch.LongTensor).to(device)
    return x, edge_index, tissue_class

def save_model(model, save_path):
    torch.save(model, save_path)
