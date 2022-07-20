from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Cell:
    label: str
    name: str
    colour: str
    alt_colour: str
    colourblind_colour: str
    id: int
    alt_id: int

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


class Organ:
    def __init__(self, cells: List[Cell], tissues: List[Tissue]):
        self.cells = cells
        self.tissues = tissues

    def cell_by_id(self, i: int):
        return self.cells[i]

    def cell_by_label(self, label):
        labels = {cell.id: cell.label for cell in self.cells}
        return self.cells[labels[label]]

    def tissue_by_label(self, label):
        labels = {tissue.label: tissue.id for tissue in self.tissues}
        return self.tissues[labels[label]]


PLACENTA = Organ(
    [
        Cell("CYT", "Cytotrophoblast", "#24ff24", "#0d8519", "#00E307", 0, 0),
        Cell("FIB", "Fibroblast", "#920000", "#7b03fc", "#C80B2A", 1, 1),
        Cell("HOF", "Hofbauer Cell", "#ffff6d", "#979903", "#FFDC3D", 2, 1),
        Cell("SYN", "Syncytiotrophoblast", "#6db6ff", "#0f0cad", "#009FFA", 3, 0),
        Cell("VEN", "Vascular Endothelial", "#ff9600", "#734c0e", "#FF6E3A", 4, 2),
        Cell("MAT", "Maternal Decidua", "#008080", "#008080", "#008169", 5, 3),
        Cell("VMY", "Vascular Myocyte", "#cc6633", "#cc6633", "#6A0213", 6, 1),
        Cell("WBC", "Leukocyte", "#2f3ec7", "#2f3ec7", "#003C86", 7, 4),
        Cell("MES", "Mesenchymal Cell", "#ff00ff", "#ff00ff", "#FF71FD", 8, 1),
        Cell("EVT", "Extra Villus Trophoblast", "#b8b0f1", "#b8b0f1", "#FFCFE2", 9, 0),
        Cell("KNT", "Syncytial Knot", "#00ffff", "#00ffff", "#7CFFFA", 10, 0),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "#000000", 0),
        Tissue("Sprout", "Villus Sprout", "#ff00ff", 1),
        Tissue("MVilli", "Mesenchymal Villi", "#ff0000", 2),
        Tissue("TVilli", "Terminal Villi", "#ff7800", 3),
        Tissue("ImIVilli", "Immature Intermediary Villi", "#a53419", 4),
        Tissue("MIVilli", "Mature Intermediary Villi", "#ffb366", 5),
        Tissue("AVilli", "Anchoring Villi", "#ffe699", 6),
        Tissue("SVilli", "Stem Villi", "#deff00", 7),
        Tissue("Chorion", "Chorionic Plate", "#669966", 8),
        Tissue("Maternal", "Basal Plate/Septum", "#53bc8d", 9),
        Tissue("Inflam", "Inflammatory Response", "#4d3399", 10),
        Tissue("Fibrin", "Fibrous Region", "#6680e6", 11),
        Tissue("Avascular", "Avascular Villi", "#6d0c67", 12),
    ],
)
PLACENTA_CORD = Organ(
    [
        Cell("EPI", "Epithelial Cell", "#ff0000", "#ff0000", "#ff0000", 0, 0),
        Cell("FIB", "Fibroblast", "#920000", "#7b03fc", "#7b03fc", 1, 1),
        Cell("MAC", "Macrophage", "#ffff6d", "#979903", "#979903", 2, 2),
        Cell("VEN", "Vascular Endothelial", "#ff9600", "#734c0e", "#734c0e", 3, 3),
        Cell("VMY", "Vascular Myocyte", "#cc6633", "#cc6633", "#cc6633", 4, 4),
        Cell("WBC", "White Blood Cell", "#2f3ec7", "#2f3ec7", "#2f3ec7", 5, 5),
        Cell("MES", "Mesenchymal Cell", "#ff00ff", "#ff00ff", "#ff00ff", 6, 6),
    ],
    [],
)
LIVER = Organ([], [])
ADIPOCYTE = Organ([], [])


def get_organ(organ_name):
    organ_dicts = {
        "placenta": PLACENTA,
        "placenta_cord": PLACENTA_CORD,
        "liver": LIVER,
        "adipocyte": ADIPOCYTE,
    }
    return organ_dicts[organ_name]
