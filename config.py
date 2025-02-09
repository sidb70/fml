from dataclasses import dataclass
import yaml
import os

@dataclass
class Config:
    device: str = "cuda"
    model_path: str = "data/topic_model"
    docs_path: str = "data/docs"
    posts_path: str = "data/posts.jsonl"
    sample_size: int = 5000

# try to load "config.yaml" file

config_path = "config.yaml"

config: Config = None

try: 
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = Config(**config_dict)
except FileNotFoundError:
    # create a new config object and save it to "config.yaml"
    config = Config()
    with open(config_path, "w") as f:
        yaml.safe_dump(config.__dict__, f)



