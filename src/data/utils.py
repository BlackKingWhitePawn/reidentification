import numpy as np
import functools


def __aggregate_to_continious(x: int | list[list[int]], y: int):
    """Аггрегирующая функция для разделения списка чисел на непрерывные отрезки"""
    if (not type(x) == list):
        if (y - x == 1):
            return [[x, y]]
        else:
            return [[x], [y]]
    else:
        last = x[-1][-1]
        if (y - last > 1):
            x.append([y])
        else:
            x[-1].append(y)
    return x


def __get_possible_tuples_count_segment(distance: int, segment: list[int]) -> int:
    """Рассчитывает количество возможных пар для отрезка"""
    return len(segment) - distance - 1


def __get_neighbours_tuples_count(distance: int, segments: list[list[int]]) -> int:
    """Рассчитывает количество возможных пар из граничных элементов"""
    sum = 0
    prev = None
    for s in segments:
        if (prev == None):
            prev = s
            continue
        if (s[0] - prev[-1] - 1 == distance):
            sum += 1
        prev = s

    return sum


def get_possible_tuples_count(distance: int, segments: list[list[int]]) -> int:
    """Рассчитывает количество возможных пар для списка отрезков"""
    sum = 0
    for s in segments:
        sum += max(0, __get_possible_tuples_count_segment(distance, s))

    sum += __get_neighbours_tuples_count(distance, segments)
    return sum


def split_to_continuous_segments(array_numbers: list[int]) -> list[list[int]]:
    """Возвращает список непрерывных отрезков чисел"""
    if (len(array_numbers) == 0):
        return [[]]
    elif (len(array_numbers) == 1):
        return [array_numbers]
    else:
        return functools.reduce(__aggregate_to_continious, sorted(array_numbers))


def _get_possible_tuples(distance: int, segment: list[int]) -> list[tuple[int, int]]:
    end = max(len(segment) - distance - 1, 0)
    return [(i, i + distance + 1) for i in range(segment[0], segment[end])]


def get_possible_tuples(distance: int, segments: list[list[int]]) -> list[tuple[int, int]]:
    """Возвращает список возможных пар чисел с заданным расстоянием для отрезка
    ### Parameters: 
    - distance: int - расстояние между элементами
    - segments: list[list[int]] - список непрерывных отрезков
    """
    res = []
    prev = None
    for segment in segments:
        tuples = _get_possible_tuples(distance, segment)
        if (prev is not None and segment[0] - prev[-1] - 1 == distance):
            res.append((prev[-1], segment[0]))

        prev = segment
        res += tuples

    return res
