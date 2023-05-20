"""
Модуль, предзначенный для обучения моделей и интепретации результатов обучения

### Внутренние модули:
- loader - функции создания классов даталоадеров и датасетов 
из известных наборов данных. Доступные датасеты для загрузки:
    - MOT20_ext (подробнее о параметрах в src.data.mot_ext)
- loss - классы для функций потерь
- train - функции для обучения моделей разной архитектуры
- utils - утилиты для обучения (отображение данных, расчет статистик, сохранение результатов)

### Настройка обучения:

1. Задать параметры датасета (config) в формате словаря со следующими значениями:
- dataset_config - название настройки. Используется для сохранения результатов обучения
- dataset - имя исходного датасета. Имена датасетов описаны в src.data
- dataset_use - доля датасета исапользуемая в обучении
- train_proportion - доля обучающей выборки
- val_proportion - доля валидационной выборки
- test_proportion - доля тестовой выборки
- batch_size - размер батча для обучения
- extra_parameters - словарь с дополнительными настройками, 
передаются параметрами в конструктор соответствующего класса Dataset.
Передавать их необходимо в формате, в котором они принимаются конструктором\n
Пример:\n
`{"dataset_config": 'mot20_ext_v1',"dataset": 'MOT20_ext',"dataset_use": 0.002,"train_proportion": 0.65,"val_proportion": 0.15,"test_proportion": 0.2 }`

2. Загрузить тренировочные наборы. Передать конфиг в фуцнкцию загрузки даталоадеров.
Пример:\n
`train_loader, val_loader, test_loader = get_loaders(current_config, transform=transform)`

3. Запустить обучение с соответствующим конфигом. 
Параметры обучения сохранятся в файл данных экспериментов
"""

from .loader import CONFIGS, get_dataset, get_loaders
from .loss import ContrastiveLoss
from .train import fit, train_siamese, _get_loss_name
from .utils import (display_batch, display_images, get_config, get_statistics,
                    save_train_results, get_experiments, )

__all__ = [
    CONFIGS,
    ContrastiveLoss,
    display_batch,
    display_images,
    fit,
    get_config,
    get_dataset,
    get_loaders,
    get_statistics,
    save_train_results,
    _get_loss_name,
    train_siamese,
    get_experiments
]
