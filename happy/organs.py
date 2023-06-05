from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Cell:
    label: str
    name: str
    colour: str
    alt_colour: str
    id: int
    alt_id: int
    structural_id: int

    def __str__(self):
        return f"{self.label}"


@dataclass(frozen=True)
class Tissue:
    label: str
    name: str
    colour: str
    id: int

    def __str__(self):
        return f"{self.label}"


@dataclass(frozen=True)
class Lesion:
    label: str
    name: str
    id: int

    def __str__(self):
        return f"{self.label}"


class Organ:
    def __init__(self, cells: List[Cell], tissues: List[Tissue], lesions: List[Lesion]):
        self.cells = cells
        self.tissues = tissues
        self.lesions = lesions

    def cell_by_id(self, i: int):
        return self.cells[i]

    def cell_by_label(self, label):
        labels = {cell.label: cell.id for cell in self.cells}
        return self.cells[labels[label]]

    def tissue_by_label(self, label):
        labels = {tissue.label: tissue.id for tissue in self.tissues}
        return self.tissues[labels[label]]


PLACENTA = Organ(
    [
        Cell("CYT", "Cytotrophoblast", "#00E307", "#0d8519", 0, 0, 2),
        Cell("FIB", "Fibroblast", "#C80B2A", "#7b03fc", 1, 1, 4),
        Cell("HOF", "Hofbauer Cell", "#FFDC3D", "#979903", 2, 1, 8),
        Cell("SYN", "Syncytiotrophoblast", "#009FFA", "#0f0cad", 3, 0, 0),
        Cell("VEN", "Vascular Endothelial", "#FF6E3A", "#734c0e", 4, 2, 6),
        Cell("MAT", "Maternal Decidua", "#008169", "#008080", 5, 3, 10),
        Cell("VMY", "Vascular Myocyte", "#6A0213", "#cc6633", 6, 1, 5),
        Cell("WBC", "Leukocyte", "#003C86", "#2f3ec7", 7, 4, 9),
        Cell("MES", "Mesenchymal Cell", "#FF71FD", "#ff00ff", 8, 1, 7),
        Cell("EVT", "Extra Villus Trophoblast", "#FFCFE2", "#b8b0f1", 9, 0, 3),
        Cell("KNT", "Syncytial Knot", "#7CFFFA", "#00ffff", 10, 0, 1),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "#000000", 0),
        Tissue("Sprout", "Villus Sprout", "#ff3cfe", 1),
        Tissue("MVilli", "Mesenchymal Villi", "#f60239", 2),
        Tissue("TVilli", "Terminal Villi", "#ff6e3a", 3),
        Tissue("ImIVilli", "Immature Intermediate Villi", "#5a000f", 4),
        Tissue("MIVilli", "Mature Intermediate Villi", "#ffac3b", 5),
        Tissue("AVilli", "Anchoring Villi", "#ffcfe2", 6),
        Tissue("SVilli", "Stem Villi", "#ffdc3d", 7),
        Tissue("Chorion", "Chorionic Plate", "#005a01", 8),
        Tissue("Maternal", "Basal Plate/Septum", "#00cba7", 9),
        Tissue("Inflam", "Inflammatory Response", "#7cfffa", 10),
        Tissue("Fibrin", "Fibrin", "#0079fa", 11),
        Tissue("Avascular", "Avascular Villi", "#450270", 12),
    ],
    [
        Lesion("healthy", "Healthy", 0),
        Lesion("infarction", "Infarction", 1),
        Lesion("perivillous_fibrin", "Perivillous Fibrin", 2),
        Lesion("intervillous_thrombos", "Intervillous Thrombos", 3),
        Lesion("avascular_villi", "Avascular Villi", 5),
        Lesion("inflammation", "Inflammation", 6),
        Lesion("edemic", "Villous Edema", 7),
        Lesion("small_villi", "Small Villi", 8),
    ],
)
PLACENTA_CORD = Organ(
    [
        Cell("EPI", "Epithelial Cell", "#ff0000", "#ff0000", 0, 0, 0),
        Cell("FIB", "Fibroblast", "#7b03fc", "#7b03fc", 1, 1, 1),
        Cell("MAC", "Macrophage", "#979903", "#979903", 2, 2, 2),
        Cell("VEN", "Vascular Endothelial", "#734c0e", "#734c0e", 3, 3, 3),
        Cell("VMY", "Vascular Myocyte", "#cc6633", "#cc6633", 4, 4, 4),
        Cell("WBC", "White Blood Cell", "#2f3ec7", "#2f3ec7", 5, 5, 5),
        Cell("MES", "Mesenchymal Cell", "#ff00ff", "#ff00ff", 6, 6, 6),
    ],
    [],
    [],
)
LIVER = Organ([], [], [])
ADIPOCYTE = Organ([], [], [])


def get_organ(organ_name):
    organ_dicts = {
        "placenta": PLACENTA,
        "placenta_cord": PLACENTA_CORD,
        "liver": LIVER,
        "adipocyte": ADIPOCYTE,
    }
    return organ_dicts[organ_name]
