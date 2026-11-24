"""
io_json.py — збереження, читання та перевірка індивідуальних задач
у форматі JSON. У файлі модулі та напрями нумеруються з 1.
"""
import json
from core.model import Problem


def problem_to_dict(p):
    return {
        "n": p.n,
        "h": p.h,
        "A": p.A,
        "L": p.L,
        "modules": [{"id": i + 1, "direction": p.direction[i] + 1,
                     "a": p.a[i], "b": p.b[i]} for i in range(p.n)],
        "directions": [{"id": t + 1, "u": p.u[t], "is_priority": t in p.P}
                       for t in range(p.h)],
        "pairs": [{"i": i + 1, "j": j + 1, "gamma": g}
                  for (i, j), g in sorted(p.pairs.items())],
    }


def dict_to_problem(d):
    n, h = d["n"], d["h"]
    a, b, direction = [0] * n, [0] * n, [0] * n
    for m in d["modules"]:
        i = m["id"] - 1
        a[i], b[i], direction[i] = m["a"], m["b"], m["direction"] - 1
    u, P = [0] * h, set()
    for dr in d["directions"]:
        t = dr["id"] - 1
        u[t] = dr["u"]
        if dr["is_priority"]:
            P.add(t)
    pairs = {}
    for pr in d["pairs"]:
        i, j = sorted((pr["i"] - 1, pr["j"] - 1))
        pairs[(i, j)] = pr["gamma"]
    return Problem(n=n, h=h, A=d["A"], a=a, b=b, direction=direction,
                   u=u, P=P, L=d["L"], pairs=pairs)


def save_problem(p, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(problem_to_dict(p), f, ensure_ascii=False, indent=2)


def load_problem(path):
    with open(path, encoding="utf-8") as f:
        return dict_to_problem(json.load(f))


def validate_problem(p):
    """Перевіряє коректність даних. Повертає (True, "ok") або (False, причина)."""
    if p.n < 1 or p.h < 1:
        return False, "n та h мають бути додатними"
    if p.A <= 0:
        return False, "ресурс A має бути додатним"
    for i in range(p.n):
        if p.a[i] <= 0:
            return False, f"витрати модуля {i + 1} мають бути додатними"
        if p.b[i] < 0:
            return False, f"базовий ефект модуля {i + 1} не може бути від'ємним"
        if not 0 <= p.direction[i] < p.h:
            return False, f"неправильний напрям модуля {i + 1}"
    for (i, j), gamma in p.pairs.items():
        if gamma <= 0:
            return False, f"бонус пари ({i + 1}, {j + 1}) має бути додатним"
    if not p.has_feasible_solution():
        return False, (f"задача не має допустимих розв'язків —\n  неможливо обрати "
                       f"L = {p.L} пріоритетних модулів у межах квот і ресурсу")
    return True, "ok"
