import argparse
from os.path import exists, isdir, join
from shutil import rmtree

from config import *
from data import MOT20Dataset, MOT20ExtDataset, MOT20Object
from data.preparing import (check_train_test_differs, extract_mot20_ext_test,
                            restore_dataset, restore_annotations)
from data.preparing import run as run_mot20_extracting


def main():
    mot20_path = join(DATA_PATH, 'MOT20')
    if not (exists(mot20_path) and isdir(mot20_path)):
        raise ValueError(
            'MOT20 data directory is not exists. Load MOT20 dataset first')

    # run_mot20_extracting(data_path=DATA_PATH)
    # if (DEBUG):
    # rmtree(join(DATA_PATH, 'MOT20_ext'), )
    # extract_mot20_ext_test(data_path=DATA_PATH, proportion=0.25)
    check_train_test_differs(data_path=DATA_PATH)
    # restore_dataset(DATA_PATH)
    # restore_annotations(DATA_PATH)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--сd', '-c', nargs=1, help="Создает датасет с указанным именем")
    # args, unknown = parser.parse_known_args()
    main()
