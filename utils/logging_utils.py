import os
import json
from datetime import datetime
import wandb

def setup_wandb(cfg):
    run = wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.name,
        config=dict(cfg),
        mode=cfg.logging.mode,
        tags=cfg.logging.tag
    )
    return run

def json_logger(logging_path):
    f = open(logging_path, 'a', buffering=1)
    def log(obj):
        obj = dict(obj)
        obj['ts'] = datetime.now().isoformat()
        f.write(json.dumps(obj) + '\n')
        f.flush()
    return log