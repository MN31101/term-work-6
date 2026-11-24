"""
generator.py — генератор індивідуальних задач (підрозділ 3.2 КР).

Параметри генератора:
  n     — кількість модулів (розмірність задачі);
  h     — кількість напрямів;
  rho   — жорсткість бюджету: A = rho * (сума всіх a_i);
  mu    — щільність пар: |W| = mu * n;
  pi    — частка пріоритетних напрямів;
  seed  — зерно генератора випадкових чисел (для відтворюваності).
"""
import random
from core.model import Problem

# Діапазони рівномірного розподілу для параметрів модулів
A_RANGE = (5, 50)        # витрати a_i
B_RANGE = (5, 30)        # базовий ефект b_i
GAMMA_RANGE = (1, 8)     # бонус пари γ_ij


def generate(n, h=5, rho=0.4, mu=1.5, pi=0.4, seed=None):
    rnd = random.Random(seed)

    # 1. Кожен модуль випадково відносимо до одного з h напрямів
    direction = [rnd.randrange(h) for _ in range(n)]

    # 2. Витрати та базові ефекти модулів
    a = [rnd.randint(*A_RANGE) for _ in range(n)]
    b = [rnd.randint(*B_RANGE) for _ in range(n)]

    # 3. Сумарний ресурс — частка rho від суми всіх витрат
    A = round(rho * sum(a), 1)

    # 4. Пари взаємопідсилення: floor(mu * n) різних пар
    max_pairs = n * (n - 1) // 2
    n_pairs = min(int(mu * n), max_pairs)
    pairs = {}
    while len(pairs) < n_pairs:
        i, j = sorted(rnd.sample(range(n), 2))
        pairs[(i, j)] = rnd.randint(*GAMMA_RANGE)

    # 5. Квоти напрямів: трохи більше за середню кількість модулів на напрям
    u = [max(2, n // h + 1) for _ in range(h)]

    # 6. Пріоритетні напрями та мінімум L
    n_priority = max(1, int(pi * h))
    P = set(rnd.sample(range(h), n_priority))
    L = max(1, int(0.3 * sum(u[t] for t in P)))

    problem = Problem(n=n, h=h, A=A, a=a, b=b, direction=direction,
                      u=u, P=P, L=L, pairs=pairs)

    # 7. Якщо задача недопустима — зменшуємо L, доки вона не стане допустимою
    while problem.L > 1 and not problem.has_feasible_solution():
        problem.L -= 1
    return problem
