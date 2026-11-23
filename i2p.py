"""Bounded perspective+caption I2P. Default is CPU preparation, NOT inference.

No checkpoint downloads: generation uses pinned, already-authorized local cache.
Run once per new output directory. Source/weights never modified.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pa_src.i2p_geometry import (intrinsics, perspective_to_sparse_erp,
                                 padded_known_mask, erp_to_perspective, rotation)

BASE = 'black-forest-labs/FLUX.1-dev'
BASE_REV = '45d95c497444bbaebe766e1528792fc9158c0c3a'
LORA = 'Insta360-Research/DiT360-Panorama-Image-Generation'
LORA_REV = '88b5915693607d3377e1b5385d7abbe6c671d6ce'
PREFIX = 'This is a panorama image. '
PARAMS = dict(height=1024, width=2048, num_inversion_steps=50,
              num_inference_steps=50, gamma=1., tau=.5, t_l=.5,
              guidance_scale=3.5, source_guidance_scale=0., source_prompt='',
              eta=1., strength=1., start_timestep=0., stop_timestep=.99,
              max_sequence_length=512, num_images_per_prompt=1, use_timestep=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def previews(erp, source, K, out, prefix):
    """Periodic sampling for all longitudes including seam, explicit pole views."""
    views = [('source reference', source), ('source view', erp_to_perspective(
        erp, K, hw=source.shape[:2]))]
    k = intrinsics(512, 512, 90)
    for name, yaw, pitch in [('right', 90, 0), ('seam', 180, 0),
                              ('left', -90, 0), ('zenith', 0, 90), ('nadir', 0, -90)]:
        views.append((name, erp_to_perspective(erp, k, rotation(yaw, pitch))))
    sheet = Image.new('RGB', (4*384, 2*326), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for i, (name, array) in enumerate(views):
        image = Image.fromarray(array)
        image.save(out / (prefix+'_'+name.replace(' ', '_')+'.png'))
        image.thumbnail((384, 288))
        x, y = (i % 4)*384, (i//4)*326
        sheet.paste(image, (x, y+30))
        draw.text((x+8, y+8), prefix+': '+name, fill='white')
    sheet.save(out / (prefix+'_contact_sheet.jpg'), quality=93)


def run_pipeline(pipe, canvas, known, prompt, seed, torch, install_processor):
    """Single sample; restore native processors even if inversion/generation fails.

    Capturing processors per call is safe because finally always restores them.
    If a future caller loops, load pipe outside the loop, never reuse RNG streams.
    """
    if pipe.vae_scale_factor != 8:
        raise ValueError('native pipeline requires VAE scale8')
    device = pipe._execution_device
    native = dict(pipe.transformer.attn_processors)
    mask = torch.from_numpy(padded_known_mask(known)).to(device)
    inversion_rng = torch.Generator(device=device).manual_seed(seed)
    generation_rng = torch.Generator(device=device).manual_seed(seed)
    try:
        pipe.transformer.set_attn_processor(dict(native))
        inv, image, ids = pipe.invert(image=Image.fromarray(canvas),
            source_prompt='', source_guidance_scale=0., num_inversion_steps=50,
            gamma=1., strength=1., height=1024, width=2048, generator=inversion_rng)
        n = mask.shape[0]
        if inv.shape[:2] != (1, n) or image.shape[:2] != (1, n) or ids.shape != (n, 3):
            raise ValueError('mask/latent/IDs token contract mismatch')
        install_processor(pipe.transformer, mask, device, n)
        return pipe([prompt, prompt], inverted_latents=inv, image_latents=image,
            latent_image_ids=ids, height=1024, width=2048,
            num_inference_steps=50, eta=1., strength=1., start_timestep=0.,
            stop_timestep=.99, guidance_scale=3.5, num_images_per_prompt=1,
            generator=generation_rng, mask=mask, use_timestep=True, t_l=.5,
            max_sequence_length=512).images[1]
    finally:
        pipe.transformer.set_attn_processor(dict(native))


def generate(canvas, known, prompt, seed):
    import torch
    from pa_src.pipeline import RFPanoInversionParallelFluxPipeline
    from pa_src.attn_processor import (PersonalizeAnythingAttnProcessor,
                                      set_flux_transformer_attn_processor)
    def install(transformer, mask, device, n):
        set_flux_transformer_attn_processor(transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, mask=mask, device=device, tau=.5, img_dims=n))
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    # Pinned cached snapshots only. Do not bypass or automatically retry HF gates.
    pipe = RFPanoInversionParallelFluxPipeline.from_pretrained(BASE,
        revision=BASE_REV, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True).to('cuda:0')
    pipe.load_lora_weights(LORA, revision=LORA_REV,
        weight_name='adapter_model.safetensors', local_files_only=True)
    load_s = time.monotonic()-start
    start = time.monotonic()
    result = run_pipeline(pipe, canvas, known, prompt, seed, torch, install)
    torch.cuda.synchronize()
    stats = dict(load_seconds=load_s, inference_seconds=time.monotonic()-start,
        allocator_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        allocator_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        sampled_nvidia_smi_peak_mib=None,
        sampled_note='External sampler required; allocator is NOT whole-device usage.',
        device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        torch_cuda=torch.version.cuda, dtype='float16',
        versions={n: importlib.metadata.version(n) for n in
            ['torch', 'diffusers', 'transformers', 'peft', 'accelerate',
             'huggingface-hub', 'safetensors', 'numpy', 'pillow']})
    return result, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--caption', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--model-files-manifest', type=Path,
                        help='Verified pinned asset SHA256 inventory from download script')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--hfov-deg', type=float)
    group.add_argument('--K-json', type=Path, help='3x3 pinhole matrix for source dimensions')
    parser.add_argument('--seed', type=int, default=42, choices=[42])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--gpu-approved', action='store_true',
                        help='Operator has coordinated GPU ownership/VRAM, not automatic reservation')
    args = parser.parse_args(argv)
    if args.generate and not args.gpu_approved:
        parser.error('generation requires explicit --gpu-approved after coordination')
    assets = None
    if args.generate:
        if args.model_files_manifest is None:
            parser.error('generation requires --model-files-manifest with verified pinned assets')
        assets = json.loads(args.model_files_manifest.read_text())
        revisions = {(row['repo'], row['revision']) for row in assets.get('files', [])}
        if assets.get('status') != 'VERIFIED' or revisions != {(BASE, BASE_REV), (LORA, LORA_REV)}:
            parser.error('asset inventory status/revisions mismatch')
    if not args.caption.strip():
        parser.error('caption must not be empty')
    source_sha = sha(args.image)
    if args.expected_sha256 and source_sha != args.expected_sha256:
        parser.error('source SHA256 mismatch')
    with Image.open(args.image) as im:
        fmt = im.format
        im.load()
        source = np.asarray(im.convert('RGB'))
    h, w = source.shape[:2]
    K = np.asarray(json.loads(args.K_json.read_text())) if args.K_json else intrinsics(w, h, args.hfov_deg)
    canvas, known = perspective_to_sparse_erp(source, K)
    mask = padded_known_mask(known)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)  # atomic refusal; never overwrite a prior run
    try:
        shutil.copyfile(args.image, out / ('source'+args.image.suffix))
        Image.fromarray(canvas).save(out / 'sparse_erp.png')
        Image.fromarray(known.astype(np.uint8)*255).save(out / 'known_mask.png')
        previews(canvas, source, K, out, 'PREPARED_NOT_GENERATED')
        code = Path(__file__).resolve().parent
        manifest = dict(status='PREPARED_NOT_GENERATED', input_path=str(args.image.resolve()),
            input_sha256=source_sha, input_format=fmt, source_wh=[w, h], caption=args.caption,
            prompt_prefix=PREFIX, effective_prompts=[PREFIX+args.caption]*2,
            intrinsics=K.tolist(), intrinsics_provenance='provided K' if args.K_json else
                'UNCALIBRATED nominal horizontal FoV; square pixels; no crop/resize',
            hfov_deg=float(np.rad2deg(2*np.arctan(w/(2*K[0, 0])))),
            vfov_deg=float(np.rad2deg(2*np.arctan(h/(2*K[1, 1])))),
            R_cam_to_erp=np.eye(3).tolist(), unknown_rgb=[128]*3,
            sampling='pixel centers; source bounds inclusive [0,W-1]/[0,H-1]; no edge extension',
            known_pixels=int(known.sum()), known_fraction=float(known.mean()),
            token_mask_shape=list(mask.shape), mask_policy='nearest white=known, row major [last,body,first]',
            params=PARAMS, rng=dict(inversion_seed=args.seed, generation_seed=args.seed,
                independent_streams=True, inversion_order=['VAE posterior', 'y_1'], device='cuda:0',
                bit_exact_cross_hardware=False),
            base=dict(repo=BASE, revision=BASE_REV), lora=dict(repo=LORA, revision=LORA_REV),
            upstream='3779fe7965473f6824994c663a0ae7a76bc7aafa',
            code_head=subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip(),
            code_sha256={str(p.relative_to(code)):sha(p) for p in [Path(__file__).resolve(),
                code/'pa_src/i2p_geometry.py', code/'pa_src/pipeline.py', code/'pa_src/attn_processor.py']},
            generation_run=False, visual_acceptance='NOT_ASSESSED',
            license_note='Base FLUX dev non-commercial; LoRA MIT does not override base. QA-training permission NOT verified.')
        write_json(out/'preparation.json', manifest)
        if args.generate:
            shutil.copyfile(args.model_files_manifest, out/'model-files.json')
            manifest['model_files_manifest_sha256'] = sha(out/'model-files.json')
            image, stats = generate(canvas, known, PREFIX+args.caption, args.seed)
            if image.size != (2048, 1024):
                raise ValueError('unexpected raw ERP size')
            image.save(out/'raw_erp.png')
            with Image.open(out/'raw_erp.png') as check:
                check.load()
                raw = np.asarray(check.convert('RGB'))
            previews(raw, source, K, out, 'GENERATED')
            manifest.update(status='GENERATED_SOFTWARE_ONLY', generation_run=True, runtime=stats,
                            raw_erp_sha256=sha(out/'raw_erp.png'))
            write_json(out/'manifest.json', manifest)
            (out/'DONE').touch(exist_ok=False)
        else:
            (out/'PREPARED').touch(exist_ok=False)
        (out/'README.zh-TW.txt').write_text(
            'DiT360 單樣本：原圖＋配對 caption\n狀態：'+manifest['status']+'\n'
            '若只有 PREPARED，這些只是幾何投影，並非模型生成；沒有 raw_erp.png。\n'
            'source 為原始 bytes；sparse_erp 中灰為未知，known_mask 白色為已知。\n'
            'FoV 預設為未校準水平80度，垂直依原始aspect換算；不是相機真值。\n'
            '不拉伸完整原圖至 ERP、不使用別家 panorama、不後貼來源像素。\n'
            '若有 raw_erp.png，仍需獨立視覺驗收，保留差結果；不表示 QA 訓練授權。\n', encoding='utf-8')
        write_json(out/'artifact-index.json', {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
        print(json.dumps(dict(output=str(out.resolve()), status=manifest['status']), ensure_ascii=False))
    except BaseException as e:
        (out/'FAILED').write_text(type(e).__name__+'\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
