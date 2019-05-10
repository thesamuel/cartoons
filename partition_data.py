from typing import Optional
import random
import json
import os
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
from collections import defaultdict

_RANDOM_SEED = 2019
IN_DATA_DIR = './data/clean-data'


def obama_trump_tag(metadata: dict) -> Optional[str]:
    all_tags = ""
    if metadata["keywords"]:
        all_tags += metadata["keywords"] + " "

    if metadata["title"]:
        all_tags += metadata["title"] + " "

    if metadata["caption"]:
        all_tags += metadata["caption"]

    all_tags = all_tags.lower()
    has_obama = "obama" in all_tags
    has_trump = "trump" in all_tags

    if has_obama and has_trump:
        return None
    elif has_obama:
        return "obama"
    elif has_trump:
        return "trump"
    else:
        return None


def partition_data(in_data_dir: str, out_data_dir: str, tag_func, balance_train: bool = False):
    if not os.path.exists(in_data_dir):
        raise NotADirectoryError(f"{in_data_dir} doesn't exist")

    if os.path.exists(out_data_dir):
        raise IsADirectoryError(f"{out_data_dir} already exists")

    in_data_dir = Path(in_data_dir)
    out_data_dir = Path(out_data_dir)

    # Get filenames from directory
    filenames = set(os.listdir(in_data_dir))
    ids = list(set(int(filename.split(os.extsep)[0]) for filename in filenames))

    # Shuffle the data
    random.seed(_RANDOM_SEED)
    random.shuffle(ids)

    # Split the data
    split_1 = int(0.8 * len(ids))
    # split_2 = int(0.9 * len(ids))
    train_ids = set(ids[:split_1])
    val_ids = set(ids[split_1:])
    # test_ids = set(ids[split_2:])

    # Keep track of classes for data balancing
    image_filenames_for_tag = defaultdict(list)

    # Copy data to output path in PyTorch's ImageFolder structure
    for cid in tqdm(ids):
        image_filename = f"{cid}.jpg"
        json_filename = f"{cid}.json"

        metadata = json.load(open(in_data_dir / json_filename))

        tag = tag_func(metadata)
        if not tag:
            continue

        # Add train or val to path
        if cid in train_ids:
            if balance_train:
                image_filenames_for_tag[tag].append(image_filename)
                continue

            out_class_dir = out_data_dir / "train" / tag
        elif cid in val_ids:
            # copy_operations["val"][tag] = image_filename
            out_class_dir = out_data_dir / "val" / tag
        # elif cid in test_ids:
        #     # copy_operations["test"][tag] = image_filename
        #     out_class_dir = out_data_dir / "test" / tag
        else:
            raise RuntimeError()

        os.makedirs(out_class_dir, exist_ok=True)
        copyfile(in_data_dir / image_filename, out_class_dir / image_filename)

    # Additional step for copying train files in a balanced way
    if balance_train:
        min_train_class_len = min([len(c) for c in image_filenames_for_tag.values()])

        for tag in image_filenames_for_tag:
            for i, image_filename in enumerate(image_filenames_for_tag[tag]):
                if i >= min_train_class_len:
                    break

                out_class_dir = out_data_dir / "train" / tag
                os.makedirs(out_class_dir, exist_ok=True)
                copyfile(in_data_dir / image_filename, out_class_dir / image_filename)


partition_data(IN_DATA_DIR, './data/balanced-classifier-data-v2/', tag_func=obama_trump_tag, balance_train=True)
