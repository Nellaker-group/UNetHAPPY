import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import DeepGraphInfomax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.models.graphsage import SAGE
from happy.models.infomax import Encoder, corruption, summary_fn
from projects.placenta.graphs.graphs.samplers.samplers import (
    PosNegNeighborSampler,
    CurriculumPosNegNeighborSampler,
    SimpleCurriculumPosNegNeighborSampler,
)


def setup_dataloader(
    model_type,
    data,
    num_layers,
    batch_size,
    num_neighbors,
    num_curriculum=0,
    simple_curriculum=False,
):
    if model_type == "graphsage":
        if num_curriculum > 0:
            if simple_curriculum:
                return SimpleCurriculumPosNegNeighborSampler(
                    num_curriculum,
                    data.edge_index,
                    sizes=[num_neighbors for _ in range(num_layers)],
                    batch_size=batch_size,
                    shuffle=True,
                    num_nodes=data.num_nodes,
                    num_workers=12,
                    return_e_id=False,
                )
            else:
                return CurriculumPosNegNeighborSampler(
                    num_curriculum,
                    data.edge_index,
                    sizes=[num_neighbors for _ in range(num_layers)],
                    batch_size=batch_size,
                    shuffle=True,
                    num_nodes=data.num_nodes,
                    num_workers=12,
                    return_e_id=False,
                    drop_last=True,
                )
        return PosNegNeighborSampler(
            data.edge_index,
            sizes=[num_neighbors for _ in range(num_layers)],
            batch_size=batch_size,
            shuffle=True,
            num_nodes=data.num_nodes,
            num_workers=12,
            return_e_id=False,
        )
    elif model_type == "infomax":
        return NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[num_neighbors for _ in range(num_layers)],
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
        )


def setup_model(model_type, data, device, layers=None, pretrained=None):
    if pretrained:
        return torch.load(pretrained / "graph_model.pt", map_location=device)
    if model_type == "graphsage":
        model = SAGE(data.num_node_features, hidden_channels=64, num_layers=layers)
    elif model_type == "infomax":
        model = DeepGraphInfomax(
            hidden_channels=512,
            encoder=Encoder(data.num_node_features, 512),
            summary=summary_fn,
            corruption=corruption,
        )
    else:
        return ValueError(f"No such model type implemented: {model_type}")
    model = model.to(device)
    return model


def setup_parameters(data, model, learning_rate, device):
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    return optimiser, x, edge_index


def train(
    model_type,
    model,
    x,
    optimiser,
    batch,
    train_loader,
    device,
    run_path,
    epoch_num,
    simple_curriculum,
):
    model.train()

    num_neg = (
        train_loader.num_negatives if hasattr(train_loader, "num_negatives") else 0
    )

    total_loss = 0
    total_examples = 0
    for batch_i, (batch_size, n_id, adjs) in enumerate(train_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimiser.zero_grad()

        if model_type == "graphsage":
            out = model(x[n_id], adjs)
            loss = _graphsage_batch(
                batch,
                out,
                batch_i,
                device,
                run_path,
                epoch_num,
                num_neg,
                simple_curriculum,
            )
            total_loss += float(loss) * batch
            total_examples += batch
        elif model_type == "infomax":
            pos_z, neg_z, summary = model(x[n_id], adjs)
            loss = _infomax_batch(model, pos_z, neg_z, summary)
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)
        else:
            raise ValueError(f"No such model type implemented: {model_type}")

        loss.backward()
        optimiser.step()
    return total_loss / total_examples


def _graphsage_batch(
    batch_size, out, batch_i, device, run_path, epoch_num, num_neg, simple_curriculum
):
    if num_neg > 0:
        if simple_curriculum:
            out, pos_out = out.split(out.size(0) // 2, dim=0)
        else:
            out, pos_out, neg_out = out.split(
                [batch_size, batch_size, batch_size * num_neg], dim=0
            )

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_similarities = torch.empty(out.size(0), device=device)
        losses_to_plot = torch.empty(10, num_neg, device=device)

        for i, node in enumerate(out):
            if simple_curriculum:
                rand_inds = torch.randint(out.size(0), (num_neg,), device=device)
                while i in rand_inds:
                    rand_inds = torch.randint(out.size(0), (num_neg,), device=device)
                negative_nodes = out[rand_inds]
            else:
                negative_nodes = neg_out[i * num_neg : (i + 1) * num_neg]

            sorted_node_similarity = (
                (-(node * negative_nodes)).sum(-1).sort(descending=True)
            )[0]

            # Store 10 negative losses to plot
            if i < 10:
                losses_to_plot[i] = sorted_node_similarity

            # pick randomly from the best 25% of options
            neg_node = sorted_node_similarity[
                torch.randint(int(num_neg * 0.25), (1,), device=device)[0]
            ]
            neg_similarities[i] = neg_node

        neg_loss = F.logsigmoid(neg_similarities)

        if batch_i == 0:
            _plot_negative_losses(run_path, epoch_num, losses_to_plot)
        return -pos_loss - neg_loss.mean()
    else:
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        return -pos_loss - neg_loss


def _infomax_batch(model, pos_z, neg_z, summary):
    return model.loss(pos_z, neg_z, summary)


def save_state(
    run_path,
    logger,
    model,
    organ_name,
    exp_name,
    run_id,
    x_min,
    y_min,
    width,
    height,
    k,
    feature,
    top_conf,
    graph_method,
    batch_size,
    num_neighbours,
    learning_rate,
    epochs,
    layers,
    num_curriculum,
):
    save_model(model, run_path / "graph_model.pt")

    params_df = pd.DataFrame(
        {
            "organ_name": organ_name,
            "exp_name": exp_name,
            "run_id": run_id,
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "k": k,
            "feature": feature.value,
            "top_conf": top_conf,
            "graph_method": graph_method.value,
            "batch_size": batch_size,
            "num_neighbours": num_neighbours,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "layers": layers,
            "num_curriculum": num_curriculum,
        },
        index=[0],
    )
    params_df.to_csv(run_path / "params.csv", index=False)

    logger.to_csv(run_path / "graph_train_stats.csv")


def save_model(model, save_path):
    torch.save(model, save_path)


def _summary_lambda(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))


def _plot_negative_losses(run_path, iteration, neg_similarities):
    all_neg_losses = F.logsigmoid(neg_similarities)
    all_neg_losses = all_neg_losses.detach().cpu().numpy()
    df = pd.DataFrame(all_neg_losses, index=list(range(0, len(all_neg_losses))))

    ax = sns.lineplot(data=df.T, legend=False, markers=True)
    ax.set(xlabel="Ranked negative nodes", ylabel="Negative Node Loss")

    save_dir = run_path / "neg_loss_plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"neg_losses_{iteration}.png"
    plt.savefig(save_path)
    plt.close()
