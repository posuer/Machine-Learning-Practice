"""Download the real pinned pretrained weights before starting the timed exercise."""
import argparse
from huggingface_hub import snapshot_download
from .common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baseline.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = snapshot_download(cfg['model_id'], revision=cfg['model_revision'],
        allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE', 'README.md'], max_workers=2)
    print(path)


if __name__ == '__main__':
    main()
