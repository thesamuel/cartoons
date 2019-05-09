from typing import Optional
import random
import json
import os
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

_IN_DATA_DIR = Path('./data/clean-data')
_RANDOM_SEED = 2019


def get_tag(metadata: dict) -> Optional[str]:
    if not metadata["keywords"]:
        return None

    keywords = metadata["keywords"].lower()
    if "obama" in keywords:
        return "obama"
    elif "trump" in keywords:
        return "trump"
    else:
        return None


def partition_data(in_data_dir, out_data_dir):
    if not os.path.exists(in_data_dir):
        raise NotADirectoryError(in_data_dir, "doesn't exist")

    if os.path.exists(out_data_dir):
        raise IsADirectoryError(out_data_dir, "already exists")

    # Get filenames from directory
    filenames = set(os.listdir(in_data_dir))
    ids = list(set(int(filename.split(os.extsep)[0]) for filename in filenames))

    # Shuffle the data
    random.seed(_RANDOM_SEED)
    random.shuffle(ids)

    # Split the data
    split_1 = int(0.8 * len(ids))
    split_2 = int(0.9 * len(ids))
    train_ids = set(ids[:split_1])
    val_ids = set(ids[split_1:split_2])
    test_ids = set(ids[split_2:])

    # Copy data to output path in PyTorch's ImageFolder structure
    for cid in tqdm(ids):
        image_filename = f"{cid}.jpg"
        json_filename = f"{cid}.json"

        with open(in_data_dir / json_filename, 'r') as f:
            metadata = json.load(f)

            tag = get_tag(metadata)
            if not tag:
                continue

            # Add train or val to path
            if cid in train_ids:
                out_path = out_data_dir / "train" / tag
            elif cid in val_ids:
                out_path = out_data_dir / "val" / tag
            elif cid in test_ids:
                out_path = out_data_dir / "test" / tag
            else:
                raise RuntimeError()

            os.makedirs(out_path, exist_ok=True)
            copyfile(in_data_dir / image_filename, out_path / image_filename)
