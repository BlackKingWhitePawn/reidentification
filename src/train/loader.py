from os.path import join

from torch import Generator
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.config import DATA_PATH
from src.data.mot_ext import MOT20ExtDataset

# конфигурации используемые в обучении
# как с этим работать - проводим обучение, если результат нам что то говорит - сохраняем конфиг
CONFIGS = {
    "mot20_ext_v1": {
        "dataset_config": 'mot20_ext_v1',
        "dataset": 'MOT20_ext',
        "dataset_use": 0.002,
        "train_proportion": 0.65,
        "val_proportion": 0.15,
        "test_proportion": 0.2,
        "batch_size": 16
    }
}


def get_dataset(config: dict, transform=None) -> Dataset | None:
    """Загружает датасет по указанному идентификатору
    ### Parameters:
    - name: str - имя датасета
        - mot20_ext - возвращает полный тренировочный набор MOT20_ext (все видео вместе)
        - mot20_ext_test - возвращает полный тестовый набор MOT20_ext (все видео вместе)
    """
    name = config['dataset']
    if (name == 'mot20_ext'):
        extra_parameters = config['extra_parameters'] if (
            'extra_parameters' in config) else {}
        dataset01 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-01/'), transform=transform, **extra_parameters)
        dataset02 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-02/'), transform=transform, **extra_parameters)
        dataset03 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-03/'), transform=transform, **extra_parameters)
        dataset05 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-05/'), transform=transform, **extra_parameters)
        return ConcatDataset([dataset01, dataset02, dataset03, dataset05])
    elif (name == 'mot20_ext_test'):
        extra_parameters = config['extra_parameters'] if (
            'extra_parameters' in config) else {}
        dataset01 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/test/MOT20-01/'), transform=transform, **extra_parameters)
        dataset02 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/test/MOT20-02/'), transform=transform, **extra_parameters)
        dataset03 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/test/MOT20-03/'), transform=transform, **extra_parameters)
        dataset05 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/test/MOT20-05/'), transform=transform, **extra_parameters)
        return ConcatDataset([dataset01, dataset02, dataset03, dataset05])


def get_loaders(config: dict, generator: Generator = None, transform=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Возвращает три лоадера train, val, test. 
    Следует использовать для загрузки датасетов с уже сохраненными параметрами.
    Для загрузки с новыми параметрами стоит использовать ручное создание лоадеров
    ### Parameters:
    - config - конфигурация датасета. Пример: \n
        `{"dataset_config": 'mot20_ext_v1',"dataset": 'MOT20_ext',"dataset_use": 0.002,"train_proportion": 0.65,"val_proportion": 0.15,"test_proportion": 0.2 }`
    - generator: Generator - опицонально, генератор для рандом сплит
    """
    dataset = get_dataset(config, transform=transform)
    dataset_use, _ = random_split(
        dataset, [config['dataset_use'], 1 - config['dataset_use']], generator=generator)
    train_proportion = config['train_proportion'] \
        if ('train_proportion' in config) else \
        1 - config['val_proportion'] - config['test_proportion']
    train_set, val_set, test_set = random_split(
        dataset_use, [
            train_proportion,
            config['val_proportion'],
            config['test_proportion']],
        generator=generator)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config['batch_size'],
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=True,
        batch_size=config['batch_size'],
        drop_last=True,
        generator=generator
    )
    test_loader = DataLoader(
        test_set,
        shuffle=True,
        batch_size=config['batch_size'],
        drop_last=True,
        generator=generator
    )

    return train_loader, val_loader, test_loader
