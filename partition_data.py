import os
import sqlite3
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(in_data_dir: str, out_data_dir: str, ids: dict, test_size: float = 0.2, balance: bool = False):
    # Check current directory structure
    if not os.path.exists(in_data_dir):
        raise NotADirectoryError(f"{in_data_dir} does not exist")
    if os.path.exists(out_data_dir):
        raise IsADirectoryError(f"{out_data_dir} already exists")
    in_data_dir = Path(in_data_dir)
    out_data_dir = Path(out_data_dir)

    # Balance data
    if balance:
        min_len = min([len(c) for c in ids.values()])
        ids = {k: np.random.choice(v, min_len, replace=False) for k, v in ids.items()}

    # Split data
    ids_split = train_test_split(*ids.values(), test_size=test_size, shuffle=True)

    # shape: (2 x num classes), where row 0 is train and row 1 is test. 
    ids_split = np.reshape(ids_split, (-1, 2)).T
    assert ids_split.shape[0] == 2

    ids_split = {stage: zip(ids.keys(), data) for stage, data in zip(["train", "val"], ids_split)}

    # Copy data to output path in PyTorch's ImageFolder structure
    for stage, data_classes in ids_split.items():
        for tag, cids in data_classes:
            image_folder_out_dir = out_data_dir / stage / tag
            os.makedirs(image_folder_out_dir, exist_ok=False)
            for cid in cids:
                image_filename = f"{cid}.jpg"
                copyfile(in_data_dir / image_filename, image_folder_out_dir / image_filename)


def obama_detector_split(in_data_dir, out_data_dir, sqlite_path):
    sql_queries = {
        "obama": """
        SELECT C.id AS id
        FROM BoundingBoxes as B
                 LEFT JOIN
             Cartoons as C
             ON B.id = C.id
        WHERE B.entity LIKE 'Obama';
        """,
        "not_obama": """
        SELECT C.id AS id
        FROM BoundingBoxes as B
                 LEFT JOIN
             Cartoons as C
             ON B.id = C.id
        WHERE B.entity IS NULL
           OR B.entity NOT LIKE 'Obama'
        """
    }

    con = sqlite3.connect(sqlite_path)
    ids = {tag: pd.read_sql_query(query, con)['id'].to_list() for tag, query in sql_queries.items()}
    split_data(in_data_dir, out_data_dir, ids, test_size=0.2, balance=True)


def main():
    obama_detector_split('data/clean-data', 'data/bounding-box-obama-detector', 'cartoons.sqlite')


if __name__ == '__main__':
    main()
