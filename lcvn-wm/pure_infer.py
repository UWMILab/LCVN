try:
    import flash_attn.flash_attn_interface as _fa

    for _old, _new in [
        ("_wrapped_flash_attn_forward", "_flash_attn_forward"),
        ("_wrapped_flash_attn_backward", "_flash_attn_backward"),
    ]:
        if not hasattr(_fa, _old) and hasattr(_fa, _new):
            setattr(_fa, _old, getattr(_fa, _new))
except ImportError:
    pass

from pathlib import Path
from typing import Optional, Tuple
import torch
from omegaconf import OmegaConf, open_dict


class LDiTPureInference:
    def __init__(
        self,
        dfot_checkpoint_path: str,
        vae_checkpoint_path: str,
        device: str = "cuda",
        use_ema: bool = True,
    ):
        self.device = device
        self.use_ema = use_ema
        self.model, self.cfg = self._load_dfot_model(dfot_checkpoint_path, use_ema=use_ema)
        self.vae = self._load_vae(vae_checkpoint_path)
        print("\n" + "=" * 50)
        print("LDiT 4-to-1 Inferencer ready.")
        print("=" * 50 + "\n")

    def _resolve_algo_cfg(self, checkpoint_path: str, checkpoint: dict):
        ckpt_path = Path(checkpoint_path).resolve()
        current = ckpt_path.parent
        for _ in range(4):
            candidate = current / ".hydra" / "config.yaml"
            if candidate.exists():
                full_cfg = OmegaConf.load(candidate)
                algo_node = getattr(full_cfg, "algorithm", full_cfg)
                print(f"Hydra config resolved: {candidate}")
                return OmegaConf.create(OmegaConf.to_container(algo_node, resolve=True))
            if current.parent == current:
                break
            current = current.parent

        hp = checkpoint.get("hyper_parameters", {})
        if isinstance(hp, dict) and hp.get("algorithm", None) is not None:
            return OmegaConf.create(hp["algorithm"])

        # Fallback: compose default LCVN DiT-XL config via Hydra
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        config_dir = Path(__file__).resolve().parent / "configurations"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            full_cfg = compose(
                config_name="config",
                overrides=[
                    "dataset=lcvn",
                    "algorithm=ldit_video_social",
                    "experiment=video_generation",
                ],
            )
            algo_cfg = OmegaConf.create(
                OmegaConf.to_container(full_cfg.algorithm, resolve=True)
            )
        print(f"Hydra config composed from defaults: {config_dir}")
        return algo_cfg

    def _load_dfot_model(self, checkpoint_path: str, use_ema: bool = True):
        print(f"Loading model weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        algo_cfg = self._resolve_algo_cfg(checkpoint_path, checkpoint)

        # Pure inference does not need torch.compile or heavy video metric models
        with open_dict(algo_cfg):
            algo_cfg.compile = False
            if "tasks" in algo_cfg:
                if "prediction" in algo_cfg.tasks:
                    algo_cfg.tasks.prediction.enabled = False
                if "interpolation" in algo_cfg.tasks:
                    algo_cfg.tasks.interpolation.enabled = False

        from algorithms.ldit.ldit_video_social import LDiTVideoSocial

        model = LDiTVideoSocial(algo_cfg)

        if (
            use_ema
            and isinstance(checkpoint, dict)
            and checkpoint.get("optimizer_states")
            and checkpoint["optimizer_states"][0].get("ema")
        ):
            model._load_ema_weights_to_state_dict(checkpoint)
            print("Loaded EMA weights from checkpoint optimizer_states.")

        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for prefix in ("algorithm.", "module."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            if new_k.startswith("diffusion_model._orig_mod."):
                new_k = new_k.replace("diffusion_model._orig_mod.", "diffusion_model.", 1)
            new_state_dict[new_k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        missing_required = [k for k in missing if model._should_include_in_checkpoint(k)]
        if missing_required or unexpected:
            raise RuntimeError(
                f"Checkpoint weight mismatch for {checkpoint_path}: "
                f"missing_required={missing_required[:10]} (total {len(missing_required)}), "
                f"unexpected={unexpected[:10]} (total {len(unexpected)})"
            )

        print(f"Weights aligned successfully ({len(new_state_dict)} tensors loaded).")
        model.eval().to(self.device)
        return model, algo_cfg

    def _load_vae(self, vae_checkpoint_path: str):
        print(f"Loading VAE decoder: {vae_checkpoint_path}")
        from algorithms.vae.image_vae.trainer import ImageVAE

        vae = ImageVAE._from_pretrained_custom(vae_checkpoint_path)
        return vae.eval().to(self.device)

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_param = next(self.vae.parameters())
        z = latents.to(device=vae_param.device, dtype=vae_param.dtype)
        if z.ndim == 5:
            b, t, c, h, w = z.shape
            decoded = self.vae.decode(z.reshape(b * t, c, h, w))
            decoded = decoded.reshape(b, t, *decoded.shape[1:])
        else:
            decoded = self.vae.decode(z)
        return torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

    @torch.no_grad()
    def predict(
        self,
        history_latents: torch.Tensor,
        actions: torch.Tensor,
        return_rgb: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        history_latents = history_latents.to(device=self.device, dtype=torch.float32)
        actions = actions.to(device=self.device, dtype=torch.float32)
        b, t = history_latents.shape[:2]

        latents_norm = self.model._normalize_x(history_latents)
        zero_pad = torch.zeros(b, 1, actions.shape[-1], device=self.device, dtype=actions.dtype)
        actions_padded = torch.cat([zero_pad, actions], dim=1)

        pred_seq = self.model._predict_sequence(
            context=latents_norm, length=t + 1, conditions=actions_padded
        )[0]
        pred_latent = self.model._unnormalize_x(pred_seq[:, -1])

        rgb = self._decode_latents(pred_latent) if return_rgb else None
        return pred_latent, rgb
