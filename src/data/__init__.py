"""
Модуль, содержащий классы Dataset для различных наборов данных, 
функции для обработки данных и подготовки датасетов и утилиты для работы с датасетами

### Доступные датасеты:
- mot - оригинальный датасет MOT20
- mot_ext - датасет пар изображений MOT20_ext, полученный из датасета MOT20 
"""

from .mot_ext import MOT20ExtDataset
from .mot import MOT20Dataset, MOT20Object

__all__ = [MOT20ExtDataset, MOT20Object, MOT20Dataset]
