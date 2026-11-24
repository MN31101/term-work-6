"""
exact.py — точні оцінки для перевірки якості алгоритмів:
  brute_force    — точний розв'язок повним перебором (лише малі n);
  lp_upper_bound — верхня межа ЦФ через LP-релаксацію.
"""
import time
from itertools import combinations
from core.model import Solution

BRUTE_FORCE_MAX_N = 20


def brute_force(p):
    """Перебирає всі 2^n підмножин модулів і повертає найкращу допустиму."""
    if p.n > BRUTE_FORCE_MAX_N:
        return None
    start = time.perf_counter()
    best_S, best_F = set(), -1
    for size in range(p.n + 1):
        for combo in combinations(range(p.n), size):
            S = set(combo)
            if p.is_feasible(S):
                F = p.objective(S)
                if F > best_F:
                    best_S, best_F = S, F
    return Solution(S=best_S, F=best_F, time=time.perf_counter() - start)


def lp_upper_bound(p):
    """Верхня межа F_LP значення ЦФ.

    Добуток x_i * x_j замінюємо новою змінною y_ij з умовами
    y_ij <= x_i, y_ij <= x_j, а булеві змінні — неперервними з [0; 1].
    Отримуємо задачу лінійного програмування; її оптимум не менший
    за оптимум вихідної задачі, тобто є верхньою межею.
    """
    from scipy.optimize import linprog

    pair_list = list(p.pairs.keys())
    n, m = p.n, len(pair_list)
    size = n + m                         # змінні: x_0..x_{n-1}, y_0..y_{m-1}

    # linprog шукає мінімум, тому беремо ЦФ з мінусом
    c = [-p.b[i] for i in range(n)] + [-p.pairs[pair] for pair in pair_list]

    rows, rhs = [], []

    # ресурс: сума a_i x_i <= A
    rows.append([p.a[i] for i in range(n)] + [0] * m)
    rhs.append(p.A)

    # квоти: сума x_i по напряму t <= u_t
    for t in range(p.h):
        rows.append([1 if p.direction[i] == t else 0 for i in range(n)] + [0] * m)
        rhs.append(p.u[t])

    # пріоритетні: сума x_i по пріоритетних >= L  (множимо на -1)
    rows.append([-1 if p.is_priority(i) else 0 for i in range(n)] + [0] * m)
    rhs.append(-p.L)

    # y_ij <= x_i  та  y_ij <= x_j
    for k, (i, j) in enumerate(pair_list):
        for x in (i, j):
            row = [0] * size
            row[n + k] = 1
            row[x] = -1
            rows.append(row)
            rhs.append(0)

    result = linprog(c, A_ub=rows, b_ub=rhs, bounds=[(0, 1)] * size,
                     method="highs")
    return -result.fun
