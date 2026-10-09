from pathlib import Path
from tqdm import tqdm
import time

import torch
import pytorch_lightning as pl
import hydra

from GaussianLSS.common import setup_config, setup_experiment, load_backbone


def setup(cfg):
    cfg.loader.val_batch_size = 1

    if "device" not in cfg:
        cfg.device = "cuda"


@hydra.main(version_base="1.3", config_path=str(Path.cwd() / "config"), config_name="config.yaml")
def main(cfg):

    setup_config(cfg, setup)

    if cfg.experiment.get("seed"):
        pl.seed_everything(cfg.experiment.seed, workers=True)

    ckpt_path = Path(cfg.ckpt)

    assert ckpt_path.exists(), \
        f"[Error] Checkpoint not found at {ckpt_path}"

    # Stesso setup dell'evaluate.py
    model_module, data_module, _ = setup_experiment(cfg)

    model_module.backbone = load_backbone(str(ckpt_path))

    # Benchmarkiamo la rete in sè
    network = model_module.backbone

    device = torch.device(cfg.device)

    network = network.to(device)
    network.eval()

    # Validation loader, batch size = 1
    loader = data_module.val_dataloader()

    sample = next(iter(loader))

    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor)
        else v
        for k, v in sample.items()
    }

    with torch.no_grad():

        latency = []

        for _ in tqdm(range(1050)):

            torch.cuda.synchronize()

            start_time = time.perf_counter()

            network(batch)  # Forward della rete: ciò che ci interessa 

            torch.cuda.synchronize()

            end_time = time.perf_counter()

            latency.append(end_time - start_time)

        # primi 50 = warm-up
        latency = latency[50:]

    latency = torch.tensor(latency).mean().item()

    print(
        "inference latency: {:.3f} ms, speed: {:.3f} fps".format(
            latency * 1000,
            1.0 / latency
        )
    )


if __name__ == "__main__":
    main()