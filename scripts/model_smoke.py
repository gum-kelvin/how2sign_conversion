"""One real GPU forward/backward/update, with no downloads or saved weights."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default=os.environ.get('VIPOSLT_REPO'))
    parser.add_argument('--output-root', default=os.environ.get('VIPOSLT_FEATURE_ROOT'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args(argv)
    if not args.repo_root or not args.output_root:
        parser.error('--repo-root and --output-root (or their environment variables) are required')
    repo = Path(os.path.expandvars(args.repo_root)).expanduser().resolve()
    output = Path(os.path.expandvars(args.output_root)).expanduser().resolve()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    source = json.loads((output/'source_manifest.json').read_text(encoding='utf-8-sig'))
    if source.get('synthetic'):
        parser.error('This real-model smoke requires real extracted data, not synthetic fixtures')
    config = json.loads((output/'configs/how2sign_edm.yaml').read_text(encoding='utf-8-sig'))
    info = json.loads((output/'configs/How2Sign.yaml').read_text(encoding='utf-8-sig'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        parser.error(f'Install the repository training dependencies on the GPU server: {exc}')
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        parser.error('A CUDA device is required for this real training smoke')
    sys.path.insert(0, str(repo))
    from dataset.loader_ft import PrecomputedFeatureDataset
    from models.edm_ddp import EDM
    model_args = dict(config['llm_args'])
    for key in ('llm', 'include_sp', 'include_pose', 'include_mo', 'sp_hidden_size', 'mo_hidden_size', 'pose_hidden_size', 'contrastive', 'include_ctc'):
        model_args[key] = config[key]
    model_args.update(dataset='How2Sign', lang='English', gloss_dict={}, pose_encoder_cfg={},
                      num_heads=config['temporal_args']['num_heads'], num_layers=config['temporal_args']['num_layers'])
    model_args.update({key: config[key] for key in ('llm_precision','gradient_checkpointing','ctc_weight') if key in config})
    if config['include_pose'] or config['include_ctc']:
        parser.error('This baseline supports spatial/motion features with CTC and pose disabled')
    previous = Path.cwd()
    torch.cuda.set_device(args.device)
    with tempfile.TemporaryDirectory(prefix='how2sign-model-smoke-') as temp:
        temp = Path(temp)
        try:
            # Loader annotation lookup is intentionally repo-relative; isolate it
            # for a pilot without replacing the user's installed corpus manifests.
            shutil.copytree(output/'preprocess/How2Sign', temp/'preprocess/How2Sign')
            os.chdir(temp)
            dataset = PrecomputedFeatureDataset(str(Path(info['feature_dir'])/'train'),
                str(Path(info['sp_path'])/'train'), info['pose_path'], str(Path(info['mo_path'])/'train'),
                dataset='How2Sign', mode='train', include_sp=config['include_sp'],
                include_pose=False, include_mo=config['include_mo'], strict=True)
            batch = dataset.collate_fn([dataset[i] for i in range(min(args.batch_size,len(dataset)))])
            os.chdir(repo)  # Existing model cache contract: ./data/models.
            backend = 'nccl' if dist.is_nccl_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method=(temp/'rendezvous').as_uri(), rank=0, world_size=1)
            model = EDM(model_args).to(args.device).train()
            sp, pose, mo, gloss, text, english, sp_len, pose_len, mo_len = batch
            move = lambda value: value.to(args.device) if value is not None else None
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
            optimizer.zero_grad(set_to_none=True)
            loss = model(move(sp), pose, move(mo), move(sp_len), move(pose_len), move(mo_len), gloss, text, english, warmup=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss: {loss}')
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError('Missing or nonfinite gradients')
            optimizer.step()
            print(json.dumps({'status':'PASS','verification':'GPU-tested','batch_ids':dataset.files[:len(text)],
                              'loss':loss.detach().cpu().item(),'finite_gradients':True,'optimizer_steps':1}))
        except OSError as exc:
            raise RuntimeError(f'Model assets must already be cached in the Hugging Face cache and {repo}/data/models. Offline loading failed: {exc}') from exc
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            os.chdir(previous)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Model smoke failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
