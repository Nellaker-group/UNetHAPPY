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
    sup_graphsaint_rw = "sup_graphsaint_rw"
    sup_graphsaint_edge = "sup_graphsaint_edge"
    sup_graphsaint_node = "sup_graphsaint_node"
    sup_shadow = "sup_shadow"
    sup_gatv2 = "sup_gatv2"
    sup_sign = "sup_sign"
    sup_sgc = "sup_sgc"
    sup_mlp = "sup_mlp"