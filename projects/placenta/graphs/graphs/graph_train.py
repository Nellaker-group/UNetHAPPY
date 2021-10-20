import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import DeepGraphInfomax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    simple_curriculum,
    tissue_class,
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
                n_id,
                num_neg,
                simple_curriculum,
                tissue_class,
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
    batch_size, out, batch_i, device, n_id, num_neg, simple_curriculum, tissue_class
):
    if num_neg > 0:
        if simple_curriculum:
            out, pos_out = out.split(out.size(0) // 2, dim=0)
            out_ids, _ = n_id[: out.size(0) + pos_out.size(0)].split(out.size(0), dim=0)

            rand_out_inds = torch.randint(
                out.size(0), (out.size(0) * num_neg,), device=device
            )
            node_inds = torch.arange(start=0, end=out.size(0), device=device)
            # Make sure the other outs used as negatives do not match the current out
            matching_inds = torch.where(
                torch.eq(node_inds.repeat_interleave(num_neg), rand_out_inds), 1, 0
            )
            if ~torch.eq(matching_inds.sum(), 0):
                bad_inds = torch.nonzero(matching_inds).reshape(-1)
                rand_out_inds[bad_inds] = rand_out_inds[bad_inds] + 1
                # If the last possible element was 'bad', find it and -2 from it
                values_that_are_too_big = torch.where(
                    torch.eq(rand_out_inds, out.size(0)), 1, 0
                )
                final_bad_inds = torch.nonzero(values_that_are_too_big).reshape(-1)
                rand_out_inds[final_bad_inds] = (torch.sub(rand_out_inds, 2))[
                    final_bad_inds
                ]
            neg_out = out[rand_out_inds]
            neg_out = torch.reshape(neg_out, (out.size(0), num_neg, 64))
            all_neg_similarities, sort_inds = (
                (-(neg_out * out[:, None])).sum(-1).sort(descending=True)
            )
            neg_ids = out_ids[rand_out_inds]

        else:
            out, pos_out, neg_out = out.split(
                [batch_size, batch_size, batch_size * num_neg], dim=0
            )
            out_ids, _, neg_ids = n_id.split[: batch_size * 2 + batch_size * num_neg](
                [batch_size, batch_size, batch_size * num_neg], dim=0
            )
            neg_out = torch.reshape(neg_out, (out.size(0), num_neg, 64))
            all_neg_similarities, sort_inds = (
                (-(neg_out * out[:, None])).sum(-1).sort(descending=True)
            )

        # pick randomly from the best 25% of options
        rand_inds = torch.randint(int(num_neg * 0.25), (out.size(0),), device=device)
        inds_multiplication = torch.arange(
            start=1, end=out.size(0) * num_neg, step=num_neg, device=device
        )
        chosen_negative_inds = rand_inds + inds_multiplication
        neg_similarities = torch.flatten(all_neg_similarities)[chosen_negative_inds]

        neg_loss = F.logsigmoid(neg_similarities).mean()
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()

        # Store 10 sorted negative losses to plot per batch
        if batch_i == 0:
            num_lines = 1

            # ids of possible negatives
            neg_ids = torch.reshape(neg_ids, (out.size(0), num_neg))
            neg_ids = torch.gather(neg_ids, dim=1, index=sort_inds)

            negative_ids_to_plot = neg_ids[:num_lines]
            target_node_ids_to_plot = out_ids[:num_lines]
            similarities_to_plot = all_neg_similarities[:num_lines]
            _plot_negative_losses(
                similarities_to_plot,
                negative_ids_to_plot,
                target_node_ids_to_plot,
                tissue_class,
            )
        return -pos_loss - neg_loss
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


def _plot_negative_losses(
    neg_similarities, negative_ids_to_plot, target_node_ids_to_plot, tissue_class
):
    all_neg_losses = F.logsigmoid(neg_similarities)
    all_neg_losses = all_neg_losses.detach().cpu().numpy()

    target_classes = tissue_class[target_node_ids_to_plot.cpu().numpy()]
    negative_classes = tissue_class[negative_ids_to_plot.cpu().numpy()]

    df = pd.DataFrame(all_neg_losses, index=list(range(0, len(all_neg_losses))))

    normalise_colours = np.linspace(0, 255, 10, dtype=int)

    palette = [
        sns.color_palette("Spectral", as_cmap=True)(normalise_colours[target_class])
        for target_class in target_classes
    ]

    ax = sns.lineplot(data=df.T, markers=True, palette=palette)
    ax.set(xlabel="Ranked negative nodes", ylabel="Negative Node Loss")
    plt.legend(title="Anchor", labels=target_classes)

    for i in range(len(df)):
        for x in range(len(df.T)):
            y = df.T[i][x]
            plt.text(
                x=x,
                y=y,
                s=negative_classes[i][x],
                color=sns.color_palette("Spectral", as_cmap=True)(
                    normalise_colours[negative_classes[i][x]]
                ),
                fontsize="large",
                fontweight="bold",
            )
