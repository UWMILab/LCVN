"""
Convert Diffusers `stabilityai/sd-vae-ft-ema` (AutoencoderKL) weights into an
`ImageVAE`-compatible checkpoint (`outputs/vae.ckpt`) with a resolved OmegaConf
`cfg`, and verify strict 1-to-1 parameter & numerical equivalence.
"""

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

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict

import torch
from diffusers import AutoencoderKL
from omegaconf import OmegaConf

WM_ROOT = Path(__file__).resolve().parents[1]
if str(WM_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_ROOT))

from algorithms.vae.image_vae.trainer import ImageVAE


def build_image_vae_cfg(resolution: int = 256):
    return OmegaConf.create(
        {
            "embed_dim": 4,
            "ddconfig": {
                "double_z": True,
                "z_channels": 4,
                "resolution": int(resolution),
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
        }
    )


def map_diffusers_key_to_image_vae(key: str, tensor: torch.Tensor):
    new_key = key

    # Top-level norm_out rename
    new_key = new_key.replace(".conv_norm_out.", ".norm_out.")
    # Residual shortcut rename
    new_key = new_key.replace(".conv_shortcut.", ".nin_shortcut.")

    # Encoder down_blocks.{i} -> encoder.down.{i}
    m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", new_key)
    if m:
        return f"encoder.down.{m.group(1)}.block.{m.group(2)}.{m.group(3)}", tensor
    m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.+)$", new_key)
    if m:
        return f"encoder.down.{m.group(1)}.downsample.{m.group(2)}", tensor

    # Decoder up_blocks.{i} -> decoder.up.{3 - i} (reversed order in CompVis/ImageVAE)
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", new_key)
    if m:
        idx = 3 - int(m.group(1))
        return f"decoder.up.{idx}.block.{m.group(2)}.{m.group(3)}", tensor
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", new_key)
    if m:
        idx = 3 - int(m.group(1))
        return f"decoder.up.{idx}.upsample.{m.group(2)}", tensor

    # Mid block resnets.{0,1} -> mid.block_{1,2}
    m = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", new_key)
    if m:
        block_num = int(m.group(2)) + 1
        return f"{m.group(1)}.mid.block_{block_num}.{m.group(3)}", tensor

    # Mid block attention
    m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.group_norm\.(.+)$", new_key)
    if m:
        return f"{m.group(1)}.mid.attn_1.norm.{m.group(2)}", tensor

    m = re.match(
        r"^(encoder|decoder)\.mid_block\.attentions\.0\.(to_q|to_k|to_v|to_out\.0)\.(weight|bias)$",
        new_key,
    )
    if m:
        side, proj, param = m.group(1), m.group(2), m.group(3)
        proj_map = {
            "to_q": "q",
            "to_k": "k",
            "to_v": "v",
            "to_out.0": "proj_out",
        }
        mapped_tensor = (
            tensor.unsqueeze(-1).unsqueeze(-1)
            if (param == "weight" and tensor.ndim == 2)
            else tensor
        )
        return f"{side}.mid.attn_1.{proj_map[proj]}.{param}", mapped_tensor

    return new_key, tensor


def convert_state_dict(diffusers_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in diffusers_sd.items():
        new_k, new_v = map_diffusers_key_to_image_vae(k, v.detach().cpu().contiguous())
        if new_k in mapped:
            raise ValueError(f"Duplicate mapped key: {new_k} (from {k})")
        mapped[new_k] = new_v
    return mapped


def verify_equivalence(
    sd_vae: AutoencoderKL,
    image_vae: ImageVAE,
    device: torch.device,
    resolution: int = 256,
) -> None:
    sd_vae = sd_vae.to(device).eval()
    image_vae = image_vae.to(device).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, resolution, resolution, device=device).clamp(-1.0, 1.0)
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        # Disable TF32 during strict numerical equivalence verification (SD-VAE uses SDPA while ImageVAE uses bmm)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            sd_dist = sd_vae.encode(x).latent_dist
            im_dist = image_vae.encode(x)
            mean_diff = (sd_dist.mean - im_dist.mean).abs().max().item()
            logvar_diff = (sd_dist.logvar - im_dist.logvar).abs().max().item()
            sd_rec = sd_vae.decode(sd_dist.mode()).sample
            im_rec = image_vae.decode(im_dist.mode())
            dec_diff = (sd_rec - im_rec).abs().max().item()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    print(
        f"[Verify] max abs diff -> mean={mean_diff:.3e}, "
        f"logvar={logvar_diff:.3e}, decode={dec_diff:.3e}"
    )
    assert mean_diff < 1e-3 and logvar_diff < 1e-3 and dec_diff < 1e-3, (
        f"Numerical equivalence check failed: mean={mean_diff}, "
        f"logvar={logvar_diff}, decode={dec_diff}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument(
        "--output", type=str, default=str(WM_ROOT.parent / "outputs" / "vae.ckpt")
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading Diffusers AutoencoderKL from: {args.source}")
    sd_vae = AutoencoderKL.from_pretrained(args.source, torch_dtype=torch.float32)
    mapped_sd = convert_state_dict(sd_vae.state_dict())

    cfg = build_image_vae_cfg(resolution=args.resolution)
    image_vae = ImageVAE(cfg)
    image_vae.load_state_dict(mapped_sd, strict=True)
    print(f"Strict load passed ({len(mapped_sd)} tensors).")

    device = torch.device(args.device)
    verify_equivalence(sd_vae, image_vae, device, resolution=args.resolution)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    payload = {
        "state_dict": mapped_sd,
        "cfg": cfg,
        "source": args.source,
    }
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    print(f"Saved ImageVAE checkpoint to: {out_path}")

    # Reload smoke verification across ImageVAE._from_pretrained_custom & convert_VAE.load_vae
    reloaded_custom = ImageVAE._from_pretrained_custom(str(out_path))
    verify_equivalence(sd_vae, reloaded_custom, device, resolution=args.resolution)

    from convert_VAE import load_vae

    reloaded_convert = load_vae(str(out_path), device, use_fp16=False)
    verify_equivalence(sd_vae, reloaded_convert, device, resolution=args.resolution)
    print("All reload & numerical equivalence checks PASSED.")


if __name__ == "__main__":
    main()
