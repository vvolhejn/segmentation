from typing import Iterable, TypeVar, Callable
import numpy as np

T = TypeVar("T")


def select_best(iterable: Iterable[T], scoring_fn: Callable[[T], float]) -> T | None:
    best = None
    best_score = -np.inf

    for item in iterable:
        score = scoring_fn(item)
        if score > best_score:
            best = item
            best_score = score

    return best, best_score
