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
    sup_clustergcn_w = "sup_clustergcn_w"
    sup_jumping = "sup_jumping"
    sup_graphsaint_rw = "sup_graphsaint_rw"
    sup_graphsaint_edge = "sup_graphsaint_edge"
    sup_graphsaint_node = "sup_graphsaint_node"
    sup_sign = "sup_sign"
    sup_shadow = "sup_shadow"
    sup_gat = "sup_gat"
    sup_gatv2 = "sup_gatv2"
    sup_mlp = "sup_mlp"
