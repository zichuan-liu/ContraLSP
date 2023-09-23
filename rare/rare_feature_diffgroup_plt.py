import argparse
import os
import pickle as pkl

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from statsmodels.tsa.arima_process import ArmaProcess
from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from utils.tools import print_results
import torch.nn as nn
from tint.models import MLP, RNN
from attribution.explainers import *
from pytorch_lightning.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap


def plt_y(y, y_pert, m, true_y):
    x = np.arange(len(y))
    plt.figure(figsize=(8, 4))

    cmap1 = cm.get_cmap('YlGn', 25)
    cmap1 = cmap1(np.linspace(0, 1, 25))
    cmap = ListedColormap(cmap1)

    # y = (y-np.min(y))/np.max(y)
    # y_pert = (y_pert-np.min(y_pert))/np.max(y_pert)
    pert = y*m+(1-m)*y_pert

    plt.pcolormesh(np.arange(len(y)), [np.min(y), np.max(y)], np.array([m, m]), cmap=cmap, alpha=0.2)

    sns.lineplot(x=x, y=pert, linewidth=2, color='dimgray')
    sns.lineplot(x=x, y=true_y, linewidth=2)
    sns.lineplot(x=x, y=y, linewidth=2, style='-.')
    plt.margins(0, 0)

    plt.gca().set_axis_off()

    plt.show()


seed_everything(42)


def run_experiment(
    cv: int = 0,
    N_ex: int = 100,
    T: int = 50,
    N_features: int = 50,
    N_select: int = 5,
):
    """Run experiment.

    Args:
        cv: Do the experiment with different cv to obtain error bars.
        N_ex: Number of time series to generate.
        T: Length of each time series.
        N_features: Number of features in each time series.
        N_select: Number of features that are truly salient.
        save_dir: Directory where the results should be saved.

    Returns:
        None
    """
    # Initialize useful variables
    random_seed = cv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Generate the input data
    ar = np.array([2, 0.5, 0.2, 0.1])  # AR coefficients
    ma = np.array([2])  # MA coefficients
    data_arima = ArmaProcess(ar=ar, ma=ma).generate_sample(nsample=(N_ex, T, N_features), axis=1)
    X = torch.tensor(data_arima, device=device, dtype=torch.float32)

    # Initialize the saliency tensors
    true_saliency = torch.zeros(size=(N_ex, T, N_features), device=device, dtype=torch.int64)
    # The truly salient features are selected randomly via a random permutation
    perm = torch.randperm(N_features, device=device)
    perm2 = torch.randperm(N_features, device=device)
    true_saliency[:N_ex//2, : int(2 * T / 4), perm[:N_select]] = 1
    true_saliency[N_ex//2:, int(T / 4) : int(3 * T / 4), perm2[:N_select]] = 1
    print("==============The mean of true_saliency is", true_saliency.sum(), "==============\n" + 100 * "=")

    # The white box only depends on the truly salient features
    def f(input):
        bs, t, n = input.shape
        output = torch.zeros((bs, t, 1), device=input.device)
        output[:N_ex//2, : int(2 * T / 4), :] = (input[:N_ex//2, : int(2 * T / 4), perm[:N_select]].sum(dim=-1).unsqueeze(-1)) ** 2
        output[N_ex//2:, int(T / 4) : int(3 * T / 4), :] = (input[N_ex//2:, int(T / 4) : int(3 * T / 4), perm2[:N_select]] ** 2).sum(dim=-1).unsqueeze(-1)
        return output

    trainer = Trainer(
        max_epochs=200,
        accelerator="cpu",
        callbacks=[EarlyStopping('train_loss', patience=10, mode='min')],
    )
    mask = GateMaskNet(
        forward_func=f,
        model=nn.Sequential(
            RNN(
                input_size=X.shape[-1],
                rnn="gru",
                hidden_size=X.shape[-1],
                bidirectional=True,
            ),
            MLP([2 * X.shape[-1], X.shape[-1]]),
        ),
        lambda_1=0.1,   # 0.1 for our lambda is suitable
        lambda_2=0.1,
        optim="adam",
        lr=0.1,
    )
    explainer = GateMask(f)
    _attr = explainer.attribute(
        X,
        trainer=trainer,
        mask_net=mask,
        batch_size=N_ex,
        sigma=0.5,
    )
    gatemask_saliency = _attr.clone().detach()
    print_results(gatemask_saliency, true_saliency)
    print("==============gatemask==============")

    condition = mask.net.model(X)

    for i in range(49, 52):
        y_pert = torch.sum(condition[:, :, perm[:N_select]], dim=2).squeeze().detach().numpy()[i]
        y = torch.sum(X[:, :, perm[:N_select]], dim=2).squeeze().detach().numpy()[i]
        true_y = f(X).squeeze().detach().numpy()[i]
        m = torch.sum(_attr, dim=2).squeeze().detach().numpy()[i]
        plt_y(y, y_pert, m, true_y)

    _, ts_dim, num_dim = condition.shape
    points = condition.reshape(-1, ts_dim * num_dim).detach().numpy()
    num_cluster = 2
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(points)
    cluster_label = kmeans.labels_
    num_cluster_set = Counter(cluster_label)

    # loss of each cluster
    loss_cluster = condition.abs().mean()  # placeholder
    for i in range(num_cluster):
        if num_cluster_set[i] < 2:
            continue
        cluster_i = points[np.where(cluster_label == i)]
        distance_i = kmeans.transform(cluster_i)[:, i]
        dist_positive = th.DoubleTensor([1])
        dist_negative = th.DoubleTensor([1])
        if num_cluster_set[i] >= 250:
            num_positive = 50
        else:
            num_positive = int(num_cluster_set[i] / 5 + 1)

        # select anchor and positive
        anchor_positive = np.argpartition(distance_i, num_positive)[:(num_positive + 1)]
        # torch anchor
        representation_anc = th.from_numpy(points[anchor_positive[0]])
        # transfer 1D to 3D
        representation_anc = th.reshape(representation_anc, (1, 1, np.shape(points)[1]))

        # positive part
        for l in range(1, num_positive + 1):
            # torch positive
            representation_pos = th.from_numpy(points[anchor_positive[l]])
            # transfer 1D to 3D
            representation_pos = th.reshape(representation_pos, (1, 1, np.shape(points)[1]))
            anchor_minus_positive = representation_anc - representation_pos
            dist_positive += th.norm(anchor_minus_positive, p=1) / np.shape(points)[1]
        dist_positive = dist_positive / num_positive

        # negative part
        for k in range(num_cluster):
            dist_cluster_k_negative = th.DoubleTensor([1])
            if k == i:
                continue
            else:
                # select negative
                if num_cluster_set[k] >= 250:
                    num_negative_cluster_k = 50
                else:
                    num_negative_cluster_k = int(num_cluster_set[k] / 5 + 1)

                negative_cluster_k = random.sample(range(points[kmeans.labels_ == k][:, 0].size),
                                                   num_negative_cluster_k)
                for j in range(num_negative_cluster_k):
                    # torch negative
                    representation_neg = th.from_numpy(points[kmeans.labels_ == k][negative_cluster_k[j]])
                    # transfer 1D to 3D
                    representation_neg = th.reshape(representation_neg, (1, 1, np.shape(points)[1]))
                    anchor_minus_negative = representation_anc - representation_neg
                    dist_cluster_k_negative += th.norm(anchor_minus_negative, p=1) / np.shape(points)[1]

            dist_cluster_k_negative = dist_cluster_k_negative / num_negative_cluster_k
            dist_negative += dist_cluster_k_negative
        dist_negative = dist_negative / (num_cluster - 1)
        #  loss =  -(margin + positives - negatives)
        loss_values = th.max((-dist_positive + dist_negative - 1)[0], th.tensor(0.)) / num_cluster

        loss_cluster += loss_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", default=0, type=int)
    parser.add_argument("--print_result", default=True, type=bool)
    args = parser.parse_args()

    run_experiment(cv=args.cv)

