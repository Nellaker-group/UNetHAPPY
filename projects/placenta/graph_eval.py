import typer
import torch

from happy.models.graphsage import SAGE
from happy.utils.utils import get_device
from happy.utils.utils import get_project_dir
from happy.cells.cells import get_organ


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
):
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / exp_name
        / model_weights_dir
        / "graph_model.pt"
    )

    model = SAGE(15000, hidden_channels=64, num_layers=2)
    state_dict = torch.load(pretrained_path, map_location=device)

    # model.load_state_dict(state_dict)

    print(state_dict)
    print(model)


if __name__ == "__main__":
    typer.run(main)
