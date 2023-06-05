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
    sup_clustergin  = "sup_clustergin"
    sup_clustergine  = "sup_clustergine"
    sup_clustergcne = "sup_clustergcne"
    sup_jumping = "sup_jumping"
    sup_graphsaint = "sup_graphsaint"
    sup_sign = "sup_sign"
    sup_shadow = "sup_shadow"
    sup_gat = "sup_gat"
    sup_gatv2 = "sup_gatv2"
    sup_mlp = "sup_mlp"


class GraphClassificationModelsArg(str, Enum):
    top_k = "top_k"
