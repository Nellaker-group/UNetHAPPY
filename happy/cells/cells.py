from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    label: str
    name: str
    colour: str
    id: int

    def __str__(self):
        return f"{self.label}"


class Organ:
    def __init__(self, *cells: Cell):
        self.cells = cells

    def by_id(self, i: int):
        return self.cells[i]

    def by_label(self, label):
        labels = {cell.id: cell.label for cell in self.cells}
        return self.cells[labels[label]]


PLACENTA = Organ(
    Cell("CYT", "Cytotrophoblasts", "#24ff24", 0),
    Cell("FIB", "Fibroblast", "#920000", 1),
    Cell("HOF", "Hofbauer", "#ffff6d", 2),
    Cell("SYN", "Syncytiotrophoblast", "#6db6ff", 3),
    Cell("VEN", "Vascular Endothelial ", "#ff9600", 4),
)
LIVER = Organ()
ADIPOCYTE = Organ()


def get_organ(organ_name):
    organ_dicts = {"placenta": PLACENTA, "liver": LIVER, "adipocyte": ADIPOCYTE}
    return organ_dicts[organ_name]
