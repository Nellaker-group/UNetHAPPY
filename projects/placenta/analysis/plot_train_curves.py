import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.utils.utils import get_project_dir


def main(
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_type: str = "sup_clustergcn",
):
    project_dir = get_project_dir(project_name)

    # get run path
    run_path = (
        project_dir / "results" / "graph" / model_type / exp_name / model_weights_dir
    )
    # get loss and accuracy curves during training
    df = pd.read_csv(run_path / "graph_train_stats.csv", index_col=0)
    loss_stats = df["train_loss"]
    accuracy_stats = df[["train_accuracy", "train_inf_accuracy", "val_accuracy"]]

    ax = sns.lineplot(data=loss_stats)
    plt.savefig(run_path / "loss_curves.png")
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.close()
    plt.clf()

    ax = sns.lineplot(data=accuracy_stats)
    plt.savefig(run_path / "accuracy_curves.png")
    ax.set(xlabel="Epoch", ylabel="Accuracy")
    plt.close()
    plt.clf()

    print(f"Plots saved to {run_path}")


if __name__ == "__main__":
    typer.run(main)
