from pathlib import Path

import torch
import pytorch_lightning as pl
import hydra

from GaussianLSS.common import setup_config, setup_experiment, load_backbone


CONFIG_PATH = Path.cwd() / "config"
CONFIG_NAME = "config.yaml"


class RegionalIoU:
    def __init__(self):
        self.thresholds = torch.tensor([0.4, 0.45, 0.5])
        self.masks = None
        self.counts = {}

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.detach().sigmoid()
        target = target.detach().bool()

        if target.ndim == 3:
            target = target.unsqueeze(1)

        assert pred.shape == target.shape, (
            f"Incopatible shape: {pred.shape} vs {target.shape}"
        )

        B, C, H, W = pred.shape
        assert (H, W) == (100, 200), f"BEV: {(H, W)}"

        if self.masks is None:
            rows = torch.arange(H, device=pred.device)[:, None]
            cols = torch.arange(W, device=pred.device)[None, :]

            # GaussianLSS: ego in the lower part, forward direction in the top part
            x = (H - rows - 0.5) * 0.5
            y = (cols + 0.5) * 0.5 - 50.0

            self.masks = {
                "Complete BEV": torch.ones(
                    (H, W), device=pred.device, dtype=torch.bool
                ),
                "X 0-10 m": ((x >= 0) & (x < 10)).expand(H, W),
                "X 10-25 m": ((x >= 10) & (x < 25)).expand(H, W),
                "X 25-50 m": ((x >= 25) & (x < 50)).expand(H, W),
                "Center |Y|<=20": (y.abs() <= 20).expand(H, W),
                "Lateral |Y|>20": (y.abs() > 20).expand(H, W),
                "Close and lateral": (
                    ((x >= 0) & (x < 10)) & (y.abs() > 20)
                ).expand(H, W),
            }

            self.counts = {
                name: torch.zeros(
                    (3, len(self.thresholds)),
                    device=pred.device,
                    dtype=torch.int64
                )
                for name in self.masks
            }

        thresholds = self.thresholds.to(pred.device)
        binary_pred = pred.unsqueeze(-1) >= thresholds

        for name, mask in self.masks.items():
            p = binary_pred[:, :, mask, :]
            t = target[:, :, mask].unsqueeze(-1)

            self.counts[name][0] += (p & t).sum(dim=(0, 1, 2))
            self.counts[name][1] += (p & ~t).sum(dim=(0, 1, 2))
            self.counts[name][2] += (~p & t).sum(dim=(0, 1, 2))

    def report(self):
        print("\n========== REGION METRICS (@0.45) ==========")
        print(f"{'Region':<23} {'IoU':>8} {'Precision':>10} {'Recall':>10} {'TP':>12} {'FP':>12} {'FN':>12}")

        for name, counts in self.counts.items():
            tp, fp, fn = counts[:, 1].cpu().tolist()

            iou = tp / (tp + fp + fn + 1e-7)
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)

            print(
                f"{name:<23} "
                f"{iou:>8.4f} "
                f"{precision:>10.4f} "
                f"{recall:>10.4f} "
                f"{tp:>12d} "
                f"{fp:>12d} "
                f"{fn:>12d}"
            )


@hydra.main(
    version_base="1.3",
    config_path=str(CONFIG_PATH),
    config_name=CONFIG_NAME
)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    setup_config(cfg)

    if cfg.experiment.get("seed"):
        pl.seed_everything(cfg.experiment.seed, workers=True)

    ckpt_path = Path(cfg.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model_module, data_module, _ = setup_experiment(cfg)
    model_module.backbone = load_backbone(str(ckpt_path))

    regional = RegionalIoU()

    def forward_hook(module, inputs, output):
        batch = inputs[0]
        regional.update(output["vehicle"], batch["vehicle"])

    hook = model_module.backbone.register_forward_hook(forward_hook)

    trainer = pl.Trainer(
        logger=False,
        **cfg.trainer
    )

    try:
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=ckpt_path
        )
    finally:
        hook.remove()

    regional.report()


if __name__ == "__main__":
    main()