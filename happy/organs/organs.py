from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Cell:
    label: str
    name: str
    colour: str
    alt_colour: str
    id: int

    def __str__(self):
        return f"{self.label}"


@dataclass(frozen=True)
class Tissue:
    label: str
    name: str
    tissue_type: str
    alt_label: str
    id: int
    alt_id: int
    tissue_id: int

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
        Cell("CYT", "Cytotrophoblasts", "#24ff24", "#0d8519", 0),
        Cell("FIB", "Fibroblast", "#920000", "#7b03fc", 1),
        Cell("HOF", "Hofbauer", "#ffff6d", "#979903", 2),
        Cell("SYN", "Syncytiotrophoblast", "#6db6ff", "#0f0cad", 3),
        Cell("VEN", "Vascular Endothelial", "#ff9600", "#734c0e", 4),
        Cell("MAT", "Maternal Decidua", "#008080", "#008080", 5),
        Cell("VMY", "Vascular Myocyte", "#cc6633", "#cc6633", 6),
        Cell("WBC", "White Blood Cell", "#2f3ec7", "#2f3ec7", 7),
        Cell("MES", "Mesenchymal Cell", "#ff00ff", "#ff00ff", 8),
        Cell("EVT", "Extra Villus Trophoblast", "#b8b0f1", "#b8b0f1", 9),
        Cell("KNT", "Syncytial Knots", "#00ffff", "#00ffff", 10),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "Pathologic", "Unlabelled", 0, 0, 0),
        Tissue("MVilli", "Mesenchymal Villi", "Foetal", "SmallVilli", 1, 1, 1),
        Tissue("TVilli", "Terminal Villi", "Foetal", "SmallVilli", 2, 1, 1),
        Tissue("IVilli", "Intermediary Villi", "Foetal", "MediumVilli", 3, 2, 1),
        Tissue("AVilli", "Anchoring Villi", "Foetal", "StemVilli", 4, 3, 1),
        Tissue("SVilli", "Villi Stem", "Foetal", "StemVilli", 5, 3, 1),
        Tissue("Chorion", "Chorion", "Foetal", "StemVilli", 6, 3, 1),
        Tissue("Maternal", "Maternal Decidua", "Maternal", "Maternal", 7, 4, 2),
        Tissue("Necrose", "Necrosed Tissue", "Pathologic", "Necrose", 8, 5, 3),
        Tissue("Infection", "Infected Tissue", "Pathologic", "Infection", 9, 6, 3),
    ],
)
LIVER = Organ([], [])
ADIPOCYTE = Organ([], [])


def get_organ(organ_name):
    organ_dicts = {"placenta": PLACENTA, "liver": LIVER, "adipocyte": ADIPOCYTE}
    return organ_dicts[organ_name]
