"""
tests/test_smoke.py — швидка перевірка коректності програмного продукту.

Запуск (з кореневої папки проєкту):  python -m tests.test_smoke
"""
import os

from core.generator import generate
from core import io_json
from algorithms import greedy, genetic, annealing, exact


def run():
    checks = []

    # 1. Індивідуальна задача №1 з розділу 1 (файл data/example_task.json)
    p = io_json.load_problem(os.path.join("data", "example_task.json"))
    g, ga, sa = greedy.solve(p), genetic.solve(p, seed=1), annealing.solve(p, seed=1)
    optimum = exact.brute_force(p)
    print(f"Задача №1: ЖА F={g.F}, ГА F={ga.F}, ІВ F={sa.F}, оптимум F={optimum.F}")
    checks.append(("ЖА на задачі №1 дає F = 93 (як у 2.5.1)", g.F == 93))
    checks.append(("ГА на задачі №1 знаходить оптимум F = 99", ga.F == optimum.F == 99))
    checks.append(("ІВ на задачі №1 знаходить оптимум F = 99", sa.F == optimum.F == 99))

    # 2. Згенерована задача
    p = generate(12, h=3, seed=1)
    checks.append(("згенерована задача коректна", io_json.validate_problem(p)[0]))

    io_json.save_problem(p, "_tmp_task.json")
    same = io_json.problem_to_dict(io_json.load_problem("_tmp_task.json")) == \
        io_json.problem_to_dict(p)
    os.remove("_tmp_task.json")
    checks.append(("збереження і читання JSON без втрат", same))

    g, ga, sa = greedy.solve(p), genetic.solve(p, seed=1), annealing.solve(p, seed=1)
    optimum = exact.brute_force(p)
    F_lp = exact.lp_upper_bound(p)
    for name, sol in (("ЖА", g), ("ГА", ga), ("ІВ", sa)):
        checks.append((f"{name} дає допустимий розв'язок", p.is_feasible(sol.S)))
        checks.append((f"{name} не кращий за оптимум", sol.F <= optimum.F))
    checks.append(("ГА і ІВ не гірші за ЖА", min(ga.F, sa.F) >= g.F))
    checks.append(("оптимум не перевищує LP-межу", optimum.F <= F_lp + 1e-6))

    # 3. Допустимість розв'язків наближених алгоритмів на задачах різної розмірності
    problems = [generate(n, seed=s) for n in (30, 100) for s in range(3)]
    checks.append(("ГА завжди повертає допустимі розв'язки",
                   all(p.is_feasible(genetic.solve(p, seed=1).S) for p in problems)))
    checks.append(("ІВ завжди повертає допустимі розв'язки",
                   all(p.is_feasible(annealing.solve(p, seed=1).S) for p in problems)))

    print()
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'ПОМИЛКА'}] {name}")
    print("\nУсі перевірки пройдено." if all(ok for _, ok in checks) else "\nЄ помилки!")


if __name__ == "__main__":
    run()
