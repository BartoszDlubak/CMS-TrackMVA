import yaml
from pathlib import Path
import datetime

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_experiment_config(path): #='configs/experiment.yaml'
    cfg = load_yaml(path)
    preprocess_path = cfg['preprocess']
    model_path = cfg['model']
    
    preprocess_cfg = load_yaml(f'configs/{preprocess_path}')
    ml_cfg = load_yaml(f'configs/{model_path}')
    return preprocess_cfg, ml_cfg


def create_experiment():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_id = f'exp_{timestamp}'
    exp_dir = Path('experiments') / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_configs(exp_dir: Path, preprocess_cfg, ml_cfg, experiment_cfg):
    with open(exp_dir / 'preprocess.yaml', 'w') as f:
        yaml.safe_dump(preprocess_cfg, f)
    with open(exp_dir / 'ml.yaml', 'w') as f:
        yaml.safe_dump(ml_cfg, f)
    with open(exp_dir / 'experiment.yaml', 'w') as f:
        yaml.safe_dump(experiment_cfg, f)
        
        
def do_cfgs_match(exp_dir: Path, curr_dir: Path):
    preprocess_cfg, ml_cfg = load_experiment_config(curr_dir)
    exp_preprocess_cfg, exp_ml_cfg = load_experiment_config(exp_dir)
    if exp_preprocess_cfg != preprocess_cfg:
        return False
    if exp_ml_cfg != ml_cfg:
        return False
    return True

# def check_resume(curr_dir):
#     all_exps = Path('experiments')
#     for exp in all_exps.iterdir():
#         if do_cfgs_match
        
# check_resume()