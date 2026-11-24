"""
solver_sa.py — алгоритм імітації відпалу (ІВ).
"""
import math
import random
import time
from core.model import Solution
from algorithms.repair import repair
from algorithms import greedy

DEFAULT_T0 = 3.0         # початкова температура
DEFAULT_ALPHA = 0.97     # коефіцієнт охолодження (обґрунтовано в розділі 5 КР)
DEFAULT_T_MIN = 0.05     # кінцева температура (умова завершення)
DEFAULT_MOVES = 100      # кількість спроб переходу при кожній температурі


def fill(p, S, rnd):
    """Доповнення: додаємо у випадковому порядку модулі, що ще вміщуються
    в ресурс і квоту (щоб після обміну не лишався невикористаний ресурс)."""
    used = p.cost(S)
    count = p.count_by_direction(S)
    outside = [i for i in range(p.n) if i not in S]
    rnd.shuffle(outside)
    for i in outside:
        t = p.direction[i]
        if used + p.a[i] <= p.A and count[t] < p.u[t]:
            S.add(i)
            used += p.a[i]
            count[t] += 1
    return S


def neighbor(p, S, rnd):
    """Сусідній розв'язок: обмін одного обраного модуля на необраний,
    ремонт і доповнення."""
    candidate = set(S)
    outside = [i for i in range(p.n) if i not in S]
    if candidate:
        candidate.remove(rnd.choice(sorted(candidate)))
    if outside:
        candidate.add(rnd.choice(outside))
    candidate = repair(p, candidate)
    if candidate is None:
        return None
    return fill(p, candidate, rnd)


def solve(p, T0=DEFAULT_T0, alpha=DEFAULT_ALPHA, T_min=DEFAULT_T_MIN,
          moves=DEFAULT_MOVES, seed=None):
    start = time.perf_counter()
    rnd = random.Random(seed)

    current = greedy.solve(p).S          # старт — розв'язок ЖА
    current_F = p.objective(current)
    best, best_F = set(current), current_F      # рекорд
    history = [best_F]
    T = T0
    levels = 0

    while T > T_min:
        for _ in range(moves):
            candidate = neighbor(p, current, rnd)
            if candidate is None:               # ремонт не вдався — пропускаємо
                continue
            delta = p.objective(candidate) - current_F
            if delta >= 0 or rnd.random() < math.exp(delta / T):
                current, current_F = candidate, current_F + delta
                if current_F > best_F:
                    best, best_F = set(current), current_F
        T *= alpha                              # охолодження
        levels += 1
        history.append(best_F)

    return Solution(S=best, F=best_F, time=time.perf_counter() - start,
                    generations=levels, history=history)
