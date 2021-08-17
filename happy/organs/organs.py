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
        Cell("CYT", "Cytotrophoblasts", "#24ff24", "#0d8519", 0),
        Cell("FIB", "Fibroblast", "#920000", "#7b03fc", 1),
        Cell("HOF", "Hofbauer", "#ffff6d", "#979903", 2),
        Cell("SYN", "Syncytiotrophoblast", "#6db6ff", "#0f0cad", 3),
        Cell("VEN", "Vascular Endothelial ", "#ff9600", "#734c0e", 4),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "other", 0),
        Tissue("MVilli", "Mesenchymal Villi", "fetal", 1),
        Tissue("TVilli", "Terminal Villi", "fetal", 2),
        Tissue("IVilli", "Intermediary Villi", "fetal", 3),
        Tissue("AVilli", "Anchoring Villi", "fetal", 4),
        Tissue("SVilli", "Villi Stem", "fetal", 5),
        Tissue("Chorion", "Chorion", "fetal", 6),
        Tissue("Maternal", "Maternal Decidua", "maternal", 7),
        Tissue("Necrose", "Necrosed", "other", 8),
    ],
)
LIVER = Organ([], [])
ADIPOCYTE = Organ([], [])


def get_organ(organ_name):
    organ_dicts = {"placenta": PLACENTA, "liver": LIVER, "adipocyte": ADIPOCYTE}
    return organ_dicts[organ_name]
