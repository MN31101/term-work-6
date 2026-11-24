"""
solver_ga.py — генетичний алгоритм (підрозділ 2.3 КР).

Особина (хромосома) — множина обраних модулів S, що відповідає
бінарному рядку x довжини n (ген x_i = 1, якщо модуль i обрано).
Усі особини в популяції допустимі — це забезпечує процедура ремонту,
тому пристосованість особини дорівнює значенню ЦФ F(S).

Схема одного покоління:
  1) турнірний відбір двох батьків;
  2) рівномірне схрещення (з імовірністю p_c);
  3) бітова мутація кожного гена (з імовірністю p_m);
  4) ремонт нащадків;
  5) нова популяція — I найкращих різних особин з батьків і нащадків
     (елітизм: найкращий розв'язок ніколи не втрачається).

Умова завершення: рекорд не покращувався k поколінь поспіль
(додатково — запобіжник max_generations).
"""
import random
import time
from core.model import Solution
from algorithms.repair import repair
from algorithms import greedy

# Значення параметрів за замовчуванням (обґрунтовано в розділі 5 КР)
DEFAULT_I = 75          # розмір популяції
DEFAULT_K = 50          # кількість поколінь без покращення рекорду
DEFAULT_PC = 0.9        # імовірність схрещення
DEFAULT_TAU = 3         # розмір турніру
MAX_GENERATIONS = 2000  # запобіжник від нескінченної роботи


# ---------- генетичні оператори ----------

def random_individual(p, rnd):
    """Випадкова допустима особина: кожен модуль беремо з імовірністю 0.5,
    потім ремонтуємо."""
    S = {i for i in range(p.n) if rnd.random() < 0.5}
    return repair(p, S)


def tournament(population, scores, tau, rnd):
    """Турнірний відбір: з tau випадкових особин обираємо найкращу."""
    contestants = rnd.sample(range(len(population)), tau)
    winner = max(contestants, key=lambda idx: scores[idx])
    return population[winner]


def crossover(parent1, parent2, n, rnd):
    """Рівномірне схрещення: кожен ген нащадок 1 бере від випадкового
    батька, а нащадок 2 — від іншого."""
    child1, child2 = set(), set()
    for i in range(n):
        if rnd.random() < 0.5:
            gene1, gene2 = i in parent1, i in parent2
        else:
            gene1, gene2 = i in parent2, i in parent1
        if gene1:
            child1.add(i)
        if gene2:
            child2.add(i)
    return child1, child2


def mutate(S, n, p_m, rnd):
    """Бітова мутація: кожен ген інвертуємо з імовірністю p_m."""
    S = set(S)
    for i in range(n):
        if rnd.random() < p_m:
            if i in S:
                S.remove(i)
            else:
                S.add(i)
    return S


def select_survivors(p, candidates, I):
    """Нова популяція: I найкращих особин без повторів."""
    candidates = sorted(candidates, key=p.objective, reverse=True)
    survivors, seen = [], set()
    for S in candidates:
        key = frozenset(S)
        if key not in seen:
            seen.add(key)
            survivors.append(S)
        if len(survivors) == I:
            break
    # якщо різних особин замало — доповнюємо найкращими (з повторами)
    while len(survivors) < I:
        survivors.append(candidates[len(survivors) % len(candidates)])
    return survivors


def initial_population(p, I, rnd):
    """Початкова популяція: розв'язок жадібного алгоритму + випадкові
    допустимі особини (без повторів, якщо це можливо)."""
    population = []
    greedy_S = greedy.solve(p).S
    if p.is_feasible(greedy_S):
        population.append(greedy_S)
    attempts = 0
    while len(population) < I and attempts < 20 * I:
        attempts += 1
        S = random_individual(p, rnd)
        if S is not None and S not in population:
            population.append(S)
    while len(population) < I:           # рідкісний випадок: мало різних
        population.append(set(population[len(population) % max(1, len(population))]))
    return population


# ---------- головна функція ----------

def solve(p, I=DEFAULT_I, k=DEFAULT_K, p_c=DEFAULT_PC, p_m=None,
          tau=DEFAULT_TAU, max_generations=MAX_GENERATIONS, seed=None):
    """Розв'язує задачу генетичним алгоритмом.

    I — розмір популяції; k — кількість поколінь без покращення
    рекорду, після якої алгоритм зупиняється; p_c — імовірність
    схрещення; p_m — імовірність мутації гена (за замовчуванням 1/n);
    tau — розмір турніру; seed — зерно випадкових чисел.
    """
    start = time.perf_counter()
    rnd = random.Random(seed)
    if p_m is None:
        p_m = 1.0 / p.n

    population = initial_population(p, I, rnd)
    if not population:                   # допустимих розв'язків не знайдено
        return Solution(S=set(), F=0, time=time.perf_counter() - start)

    best = max(population, key=p.objective)     # рекордний розв'язок S*
    best_F = p.objective(best)
    history = [best_F]
    no_improve = 0
    generation = 0

    while no_improve < k and generation < max_generations:
        generation += 1
        scores = [p.objective(S) for S in population]

        # створюємо I нащадків
        children = []
        while len(children) < I:
            parent1 = tournament(population, scores, tau, rnd)
            parent2 = tournament(population, scores, tau, rnd)
            if rnd.random() < p_c:
                child1, child2 = crossover(parent1, parent2, p.n, rnd)
            else:
                child1, child2 = set(parent1), set(parent2)
            for child, parent in ((child1, parent1), (child2, parent2)):
                child = repair(p, mutate(child, p.n, p_m, rnd))
                # якщо ремонт не вдався — замість нащадка беремо копію батька
                children.append(child if child is not None else set(parent))

        # оновлюємо популяцію (елітизм)
        population = select_survivors(p, population + children, I)

        # оновлюємо рекорд
        if p.objective(population[0]) > best_F:
            best = population[0]
            best_F = p.objective(best)
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_F)

    return Solution(S=set(best), F=best_F, time=time.perf_counter() - start,
                    generations=generation, history=history)
