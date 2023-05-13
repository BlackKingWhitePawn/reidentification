from src.data.mot_ext import MOT20ExtDataset
from os.path import join
from src.config import DATA_PATH
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
from torch import Generator

# конфигурации используемые в обучении
# как с этим работать - проводим обучение, если результат нам что то говорит - сохраняем конфиг
CONFIGS = [
    {
        "dataset_config": 'mot20_ext_v1',
        "dataset": 'MOT20_ext',
        "dataset_use": 0.002,
        "train_proportion": 0.65,
        "val_proportion": 0.15,
        "test_proportion": 0.2,
        "batch_size": 16
    }
]


def get_dataset(name: str, transform: None) -> Dataset | None:
    """Загружает датасет по указанному идентификатору
    ### Parameters:
    - name: str - имя датасета
        - MOT20_ext - возвращает полный тренировочный набор MOT20_ext (все видео вместе)
    """
    if (name == 'MOT20_ext'):
        dataset01 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-01/'), transform=transform)
        dataset02 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-02/'), transform=transform)
        dataset03 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-03/'), transform=transform)
        dataset05 = MOT20ExtDataset(
            join(DATA_PATH, 'MOT20_ext/train/MOT20-05/'), transform=transform)
        return ConcatDataset([dataset01, dataset02, dataset03, dataset05])


def get_loaders(config: dict, generator: Generator = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Возвращает три лоадера train, val, test. 
    Следует использовать для загрузки датасетов с уже сохраненными параметрами.
    Для загрузки с новыми параметрами стоит использовать ручное создание лоадеров
    ### Parameters:
    - config - конфигурация датасета. Пример: \n
        `{"dataset_config": 'mot20_ext_v1',"dataset": 'MOT20_ext',"dataset_use": 0.002,"train_proportion": 0.65,"val_proportion": 0.15,"test_proportion": 0.2 }`
    - generator: Generator - опицонально, генератор для рандом сплит
    """
    dataset = get_dataset(config['dataset'])
    dataset_use, _ = random_split(
        dataset, [config['dataset_use'], 1 - config['dataset_use']], generator=generator)
    train_set, val_set, test_set = random_split(dataset_use, [
                                                config['train_proportion'], config['vrain_proportion'], config['test_proportion']], generator=generator)
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
