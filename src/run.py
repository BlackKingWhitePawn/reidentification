from data.mot import MOT20Dataset, MOT20Object
from config import *
from os.path import exists, join, isdir
from data.preparing.mot import run as run_mot20_extracting
from shutil import rmtree


def main():
    mot20_path = join(DATA_PATH, 'MOT20')
    if not (exists(mot20_path) and isdir(mot20_path)):
        raise ValueError(
            'MOT20 data directory is not exists. Load MOT20 dataset first')

    run_mot20_extracting(data_path=DATA_PATH)
    if (DEBUG):
        rmtree(join(DATA_PATH, 'MOT20_ext'), )


if __name__ == '__main__':
    main()
