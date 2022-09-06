from enum import Enum

class FeatureArg(str, Enum):
    predictions = "predictions"
    embeddings = "embeddings"


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"
    intersection = "intersection"


class SupervisedModelsArg(str, Enum):
    sup_graphsage = "sup_graphsage"
    sup_clustergcn = "sup_clustergcn"
    sup_jumping = "sup_jumping"
    sup_gat = "sup_gat"
    sup_gat_pyg = "sup_gat_pyg"
    sup_graphsaint = "sup_graphsaint"
