from dataclasses import dataclass, asdict
import json
import argparse

@dataclass
class SentimentClassificationConfig:
    model_path : str
    host: str
    port: int
    use_cuda: bool


    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    @classmethod
    def from_yaml(cls, yaml_file):
        config_yaml = yaml.load(open(yaml_file), yaml.FullLoader)
        return cls(**config_yaml)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value