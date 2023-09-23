import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn
import os
from utils.tools import print_results

from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from tint.datasets import Mimic3
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)
from tint.models import MLP, RNN

from mortality.classifier import MimicClassifierNet


def main(
    explainers: List[str],
    areas: list,
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    is_train: bool = True,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    output_file: str = "results.csv",
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    mimic3 = Mimic3(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = MimicClassifierNet(
        feature_size=31,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.getrandbits(128),
        ),
    )
    if is_train:
        trainer.fit(classifier, datamodule=mimic3)
        if not os.path.exists("./model/"):
            os.makedirs("./model/")
        th.save(classifier.state_dict(), "./model/classifier_{}_{}".format(fold, seed))
    else:
        classifier.load_state_dict(th.load("./model/classifier_{}_{}".format(fold, seed)))
    # Get data for explainers
    with lock:
        x_train = mimic3.preprocess(split="train")["x"].to(device)
        x_test = mimic3.preprocess(split="test")["x"].to(device)
        y_test = mimic3.preprocess(split="test")["y"].to(device)

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            task="binary",
            show_progress=True,
        ).abs()

    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = MaskNet(
            forward_func=classifier,
            perturbation="fade_moving_average",
            keep_ratio=list(np.arange(0.1, 0.7, 0.1)),
            deletion_mode=True,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            time_reg_factor=0.0,
            loss="cross_entropy",
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

    if "gate_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = GateMaskNet(
            forward_func=classifier,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            loss="cross_entropy",
            optim="adam",
            lr=0.01,
        )
        explainer = GateMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=x_test.shape[0],
            sigma=0.5,
        )
        attr["gate_mask"] = _attr.to(device)

    if "extremal_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = ExtremalMaskNet(
            forward_func=classifier,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            loss="cross_entropy",
            optim="adam",
            lr=0.01,
        )
        explainer = ExtremalMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask"] = _attr.to(device)

    if "fit" in explainers:
        generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
        trainer = Trainer(
            max_epochs=200,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=10,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        explainer = Fit(
            classifier,
            generator=generator,
            datamodule=mimic3,
            trainer=trainer,
            features=x_train,
        )
        attr["fit"] = explainer.attribute(x_test, show_progress=True)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier.cpu()))
        attr["gradient_shap"] = explainer.attribute(
            x_test.cpu(),
            baselines=th.cat([x_test.cpu() * 0, x_test.cpu()]),
            n_samples=50,
            stdevs=0.0001,
            task="binary",
            show_progress=True,
        ).abs()
        classifier.to(device)

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            internal_batch_size=200,
            task="binary",
            show_progress=True,
        ).abs()

    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            task="binary",
            show_progress=True,
        ).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    if "retain" in explainers:
        retain = RetainNet(
            dim_emb=128,
            dropout_emb=0.4,
            dim_alpha=8,
            dim_beta=8,
            dropout_context=0.4,
            dim_output=2,
            temporal_labels=False,
            loss="cross_entropy",
        )
        explainer = Retain(
            datamodule=mimic3,
            retain=retain,
            trainer=Trainer(
                max_epochs=50,
                accelerator=accelerator,
                devices=device_id,
                deterministic=deterministic,
                logger=TensorBoardLogger(
                    save_dir=".",
                    version=random.getrandbits(128),
                ),
            ),
        )
        attr["retain"] = (
            explainer.attribute(x_test, target=y_test).abs().to(device)
        )

    # Classifier and x_test to cpu
    classifier.to("cpu")
    x_test = x_test.to("cpu")

    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)

    # Dict for baselines
    baselines_dict = {0: "Average", 1: "Zeros"}

    with open(output_file, "a") as fp, lock:
        for i, baselines in enumerate([x_avg, 0.0]):
            for topk in areas:
                for k, v in attr.items():
                    acc = accuracy(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    comp = comprehensiveness(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    ce = cross_entropy(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    l_odds = log_odds(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    suff = sufficiency(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )

                    fp.write(str(seed) + ",")
                    fp.write(str(fold) + ",")
                    fp.write(baselines_dict[i] + ",")
                    fp.write(str(topk) + ",")
                    fp.write(k + ",")
                    fp.write(str(lambda_1) + ",")
                    fp.write(str(lambda_2) + ",")
                    fp.write(f"{acc:.4},")
                    fp.write(f"{comp:.4},")
                    fp.write(f"{ce:.4},")
                    fp.write(f"{l_odds:.4},")
                    fp.write(f"{suff:.4}")
                    fp.write("\n")

    if not os.path.exists("./results_gate/"):
        os.makedirs("./results_gate/")
    for key in attr.keys():
        result = attr[key]
        if isinstance(result, tuple): result = result[0]
        np.save('./results_gate/{}_result_{}_{}.npy'.format(key, fold, seed), result.detach().numpy())


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            # "deep_lift",
            # "dyna_mask",
            # "extremal_mask",    #1018265, mean(0.2939）
            "gate_mask",    #577485
            # "fit",
            # "gradient_shap",
            # "integrated_gradients",
            # "lime",
            # "augmented_occlusion",
            # "occlusion",
            # "retain",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train thr rnn classifier.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=0.001,   # 0.01
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=0.01,    #0.01
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results_gate.csv",
        help="Where to save the results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # main(
    #     explainers=args.explainers,
    #     areas=args.areas,
    #     device=args.device,
    #     fold=args.fold,
    #     seed=args.seed,
    #     is_train=args.train,
    #     deterministic=args.deterministic,
    #     lambda_1=args.lambda_1,
    #     lambda_2=args.lambda_2,
    #     output_file=args.output_ile,
    # )

    for i in [0,1,2,3,4]:
        main(
            explainers=["gate_mask"],
            areas=args.areas,
            device=args.device,
            fold=i,
            seed=args.seed,
            is_train=args.train,
            deterministic=args.deterministic,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            output_file=args.output_file,
        )


"""
extremal_mask
Epoch 99:  15%|█▌        | 7/46 [00:05<00:32,  1.21it/s, v_num=2.17e+38]tensor(861030.5625, grad_fn=<SumBackward0>) tensor(0.2361, grad_fn=<MeanBackward0>)
Epoch 99:  17%|█▋        | 8/46 [00:06<00:31,  1.21it/s, v_num=2.17e+38]tensor(857074.2500, grad_fn=<SumBackward0>) tensor(0.2371, grad_fn=<MeanBackward0>)
Epoch 99:  20%|█▉        | 9/46 [00:07<00:30,  1.21it/s, v_num=2.17e+38]tensor(854596.5000, grad_fn=<SumBackward0>) tensor(0.2385, grad_fn=<MeanBackward0>)
Epoch 99:  22%|██▏       | 10/46 [00:08<00:29,  1.22it/s, v_num=2.17e+38]tensor(853049., grad_fn=<SumBackward0>) tensor(0.2402, grad_fn=<MeanBackward0>)
Epoch 99:  24%|██▍       | 11/46 [00:09<00:28,  1.22it/s, v_num=2.17e+38]tensor(852350.3750, grad_fn=<SumBackward0>) tensor(0.2420, grad_fn=<MeanBackward0>)
Epoch 99:  26%|██▌       | 12/46 [00:09<00:27,  1.22it/s, v_num=2.17e+38]tensor(852395., grad_fn=<SumBackward0>) tensor(0.2440, grad_fn=<MeanBackward0>)
Epoch 99:  28%|██▊       | 13/46 [00:10<00:27,  1.22it/s, v_num=2.17e+38]tensor(852878.3750, grad_fn=<SumBackward0>) tensor(0.2460, grad_fn=<MeanBackward0>)
Epoch 99:  30%|███       | 14/46 [00:11<00:26,  1.22it/s, v_num=2.17e+38]tensor(853561.6250, grad_fn=<SumBackward0>) tensor(0.2480, grad_fn=<MeanBackward0>)
Epoch 99:  33%|███▎      | 15/46 [00:12<00:25,  1.22it/s, v_num=2.17e+38]tensor(854232.4375, grad_fn=<SumBackward0>) tensor(0.2501, grad_fn=<MeanBackward0>)
Epoch 99:  35%|███▍      | 16/46 [00:13<00:24,  1.22it/s, v_num=2.17e+38]tensor(855200.1250, grad_fn=<SumBackward0>) tensor(0.2521, grad_fn=<MeanBackward0>)
Epoch 99:  37%|███▋      | 17/46 [00:13<00:23,  1.22it/s, v_num=2.17e+38]tensor(856566.8750, grad_fn=<SumBackward0>) tensor(0.2542, grad_fn=<MeanBackward0>)
Epoch 99:  39%|███▉      | 18/46 [00:14<00:22,  1.22it/s, v_num=2.17e+38]tensor(858125.1875, grad_fn=<SumBackward0>) tensor(0.2562, grad_fn=<MeanBackward0>)
Epoch 99:  41%|████▏     | 19/46 [00:15<00:22,  1.22it/s, v_num=2.17e+38]tensor(860028.3750, grad_fn=<SumBackward0>) tensor(0.2582, grad_fn=<MeanBackward0>)
Epoch 99:  43%|████▎     | 20/46 [00:16<00:21,  1.22it/s, v_num=2.17e+38]tensor(862223.3125, grad_fn=<SumBackward0>) tensor(0.2603, grad_fn=<MeanBackward0>)
Epoch 99:  46%|████▌     | 21/46 [00:17<00:20,  1.22it/s, v_num=2.17e+38]tensor(864697.3750, grad_fn=<SumBackward0>) tensor(0.2623, grad_fn=<MeanBackward0>)
Epoch 99:  48%|████▊     | 22/46 [00:18<00:19,  1.22it/s, v_num=2.17e+38]tensor(867668.5000, grad_fn=<SumBackward0>) tensor(0.2643, grad_fn=<MeanBackward0>)
Epoch 99:  50%|█████     | 23/46 [00:18<00:18,  1.22it/s, v_num=2.17e+38]tensor(871044.6875, grad_fn=<SumBackward0>) tensor(0.2663, grad_fn=<MeanBackward0>)
Epoch 99:  52%|█████▏    | 24/46 [00:19<00:18,  1.22it/s, v_num=2.17e+38]tensor(875049.6250, grad_fn=<SumBackward0>) tensor(0.2683, grad_fn=<MeanBackward0>)
Epoch 99:  54%|█████▍    | 25/46 [00:20<00:17,  1.22it/s, v_num=2.17e+38]tensor(879938.7500, grad_fn=<SumBackward0>) tensor(0.2702, grad_fn=<MeanBackward0>)
Epoch 99:  57%|█████▋    | 26/46 [00:21<00:16,  1.22it/s, v_num=2.17e+38]tensor(884901.2500, grad_fn=<SumBackward0>) tensor(0.2721, grad_fn=<MeanBackward0>)
Epoch 99:  59%|█████▊    | 27/46 [00:22<00:15,  1.22it/s, v_num=2.17e+38]tensor(890496., grad_fn=<SumBackward0>) tensor(0.2740, grad_fn=<MeanBackward0>)
Epoch 99:  61%|██████    | 28/46 [00:22<00:14,  1.22it/s, v_num=2.17e+38]tensor(896780.9375, grad_fn=<SumBackward0>) tensor(0.2759, grad_fn=<MeanBackward0>)
Epoch 99:  63%|██████▎   | 29/46 [00:23<00:13,  1.22it/s, v_num=2.17e+38]tensor(903392.1250, grad_fn=<SumBackward0>) tensor(0.2777, grad_fn=<MeanBackward0>)
Epoch 99:  65%|██████▌   | 30/46 [00:24<00:13,  1.22it/s, v_num=2.17e+38]tensor(910877.5000, grad_fn=<SumBackward0>) tensor(0.2796, grad_fn=<MeanBackward0>)
Epoch 99:  67%|██████▋   | 31/46 [00:25<00:12,  1.22it/s, v_num=2.17e+38]tensor(919423.4375, grad_fn=<SumBackward0>) tensor(0.2814, grad_fn=<MeanBackward0>)
Epoch 99:  70%|██████▉   | 32/46 [00:26<00:11,  1.22it/s, v_num=2.17e+38]tensor(928321.5000, grad_fn=<SumBackward0>) tensor(0.2833, grad_fn=<MeanBackward0>)
Epoch 99:  72%|███████▏  | 33/46 [00:27<00:10,  1.22it/s, v_num=2.17e+38]tensor(938521.8750, grad_fn=<SumBackward0>) tensor(0.2851, grad_fn=<MeanBackward0>)
Epoch 99:  74%|███████▍  | 34/46 [00:27<00:09,  1.22it/s, v_num=2.17e+38]tensor(949629.0625, grad_fn=<SumBackward0>) tensor(0.2868, grad_fn=<MeanBackward0>)
Epoch 99:  76%|███████▌  | 35/46 [00:28<00:09,  1.21it/s, v_num=2.17e+38]tensor(961456.1875, grad_fn=<SumBackward0>) tensor(0.2886, grad_fn=<MeanBackward0>)
Epoch 99:  78%|███████▊  | 36/46 [00:29<00:08,  1.21it/s, v_num=2.17e+38]tensor(974049.3750, grad_fn=<SumBackward0>) tensor(0.2903, grad_fn=<MeanBackward0>)
Epoch 99:  80%|████████  | 37/46 [00:30<00:07,  1.21it/s, v_num=2.17e+38]tensor(987590.3125, grad_fn=<SumBackward0>) tensor(0.2921, grad_fn=<MeanBackward0>)
Epoch 99:  83%|████████▎ | 38/46 [00:31<00:06,  1.21it/s, v_num=2.17e+38]tensor(1002570.1250, grad_fn=<SumBackward0>) tensor(0.2938, grad_fn=<MeanBackward0>)
Epoch 99:  85%|████████▍ | 39/46 [00:32<00:05,  1.21it/s, v_num=2.17e+38]tensor(1018675.6250, grad_fn=<SumBackward0>) tensor(0.2955, grad_fn=<MeanBackward0>)
Epoch 99:  87%|████████▋ | 40/46 [00:32<00:04,  1.21it/s, v_num=2.17e+38]tensor(1035405.3750, grad_fn=<SumBackward0>) tensor(0.2972, grad_fn=<MeanBackward0>)
Epoch 99:  89%|████████▉ | 41/46 [00:33<00:04,  1.21it/s, v_num=2.17e+38]tensor(1053252.1250, grad_fn=<SumBackward0>) tensor(0.2989, grad_fn=<MeanBackward0>)
Epoch 99:  91%|█████████▏| 42/46 [00:34<00:03,  1.21it/s, v_num=2.17e+38]tensor(1072391.5000, grad_fn=<SumBackward0>) tensor(0.3006, grad_fn=<MeanBackward0>)
Epoch 99:  93%|█████████▎| 43/46 [00:35<00:02,  1.21it/s, v_num=2.17e+38]tensor(1092838.8750, grad_fn=<SumBackward0>) tensor(0.3022, grad_fn=<MeanBackward0>)
Epoch 99:  96%|█████████▌| 44/46 [00:36<00:01,  1.21it/s, v_num=2.17e+38]tensor(1113879.3750, grad_fn=<SumBackward0>) tensor(0.3038, grad_fn=<MeanBackward0>)
Epoch 99:  98%|█████████▊| 45/46 [00:37<00:00,  1.21it/s, v_num=2.17e+38]tensor(1136078.8750, grad_fn=<SumBackward0>) tensor(0.3055, grad_fn=<MeanBackward0>)
Epoch 99: 100%|██████████| 46/46 [00:38<00:00,  1.21it/s, v_num=2.17e+38]


Epoch 83:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1615062.2500) tensor(0.5872, grad_fn=<MeanBackward0>) tensor(0.0162, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0623, grad_fn=<NllLossBackward0>)
Epoch 84:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1611993.6250) tensor(0.5867, grad_fn=<MeanBackward0>) tensor(0.0123, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0621, grad_fn=<NllLossBackward0>)
Epoch 85:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1608233.5000) tensor(0.5861, grad_fn=<MeanBackward0>) tensor(0.0142, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 86:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1604274.5000) tensor(0.5856, grad_fn=<MeanBackward0>) tensor(0.0140, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0620, grad_fn=<NllLossBackward0>)
Epoch 87:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1603239.8750) tensor(0.5853, grad_fn=<MeanBackward0>) tensor(0.0156, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0620, grad_fn=<NllLossBackward0>)
Epoch 88:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1600785.7500) tensor(0.5848, grad_fn=<MeanBackward0>) tensor(0.0133, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 89:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1594598.7500) tensor(0.5839, grad_fn=<MeanBackward0>) tensor(0.0123, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0620, grad_fn=<NllLossBackward0>)
Epoch 90:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1589731.8750) tensor(0.5832, grad_fn=<MeanBackward0>) tensor(0.0171, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0621, grad_fn=<NllLossBackward0>)
Epoch 91:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1586420.2500) tensor(0.5827, grad_fn=<MeanBackward0>) tensor(0.0123, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 92:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1582922.) tensor(0.5821, grad_fn=<MeanBackward0>) tensor(0.0133, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 93:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1578992.7500) tensor(0.5813, grad_fn=<MeanBackward0>) tensor(0.0126, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 94:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1573776.) tensor(0.5806, grad_fn=<MeanBackward0>) tensor(0.0114, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0620, grad_fn=<NllLossBackward0>)
Epoch 95:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1569269.) tensor(0.5800, grad_fn=<MeanBackward0>) tensor(0.0119, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 96:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1565748.5000) tensor(0.5795, grad_fn=<MeanBackward0>) tensor(0.0119, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0618, grad_fn=<NllLossBackward0>)
Epoch 97:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1561592.1250) tensor(0.5788, grad_fn=<MeanBackward0>) tensor(0.0124, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 98:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1559578.5000) tensor(0.5783, grad_fn=<MeanBackward0>) tensor(0.0141, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0620, grad_fn=<NllLossBackward0>)
Epoch 99:   0%|          | 0/1 [00:00<?, ?it/s, v_num=8.42e+37]tensor(1554224.6250) tensor(0.5774, grad_fn=<MeanBackward0>) tensor(0.0130, grad_fn=<MulBackward0>) 0.1 1.0 tensor(0.0619, grad_fn=<NllLossBackward0>)
Epoch 99: 100%|██████████| 1/1 [00:16<00:00, 16.39s/it, v_num=8.42e+37]
"""