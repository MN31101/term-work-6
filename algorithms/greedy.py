"""
solver_greedy.py — жадібний алгоритм (підрозділ 2.2 КР).

Ідея: на кожному кроці додаємо модуль з найбільшим відношенням
Δ_i(S) / a_i — «приріст ЦФ на одиницю ресурсу», де Δ_i(S) враховує
бонуси за пари з уже обраними модулями.

Алгоритм має дві фази:
  фаза 1 — набираємо L модулів лише з пріоритетних напрямів
           (щоб гарантовано виконати обмеження (1.4));
  фаза 2 — додаємо будь-які модулі, доки дозволяють ресурс і квоти.
"""
import time
from core.model import Solution


def solve(p):
    start = time.perf_counter()

    S = set()               # обрані модулі
    used = 0                # витрачений ресурс
    count = [0] * p.h       # скільки модулів обрано в кожному напрямі
    chosen_priority = 0     # скільки обрано пріоритетних модулів

    def can_add(i):
        """Чи можна додати модуль i, не порушивши ресурс і квоту."""
        t = p.direction[i]
        return i not in S and used + p.a[i] <= p.A and count[t] < p.u[t]

    def best_candidate(only_priority):
        """Модуль з максимальним Δ_i(S) / a_i серед тих, що можна додати."""
        best, best_ratio = None, -1
        for i in range(p.n):
            if only_priority and not p.is_priority(i):
                continue
            if not can_add(i):
                continue
            ratio = p.gain(S, i) / p.a[i]
            if ratio > best_ratio:
                best, best_ratio = i, ratio
        return best

    # ----- Фаза 1: набираємо L пріоритетних модулів -----
    while chosen_priority < p.L:
        i = best_candidate(only_priority=True)
        if i is None:
            break           # не вдалося виконати обмеження (1.4)
        S.add(i)
        used += p.a[i]
        count[p.direction[i]] += 1
        chosen_priority += 1

    # ----- Фаза 2: додаємо будь-які модулі, поки це можливо -----
    while True:
        i = best_candidate(only_priority=False)
        if i is None:
            break
        S.add(i)
        used += p.a[i]
        count[p.direction[i]] += 1

    return Solution(S=S, F=p.objective(S), time=time.perf_counter() - start)
