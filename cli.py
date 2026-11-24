"""
cli.py — інтерфейс командного рядка програмного продукту.

Запуск:  python cli.py

Головне меню:
  1. Робота з індивідуальною задачею (ввести / згенерувати / файл / редагувати)
  2. Розв'язати задачу всіма алгоритмами
  3. Параметри наближених алгоритмів (ГА та ІВ)
  4. Провести експерименти
  5. Вивести дані задачі
  0. Завершити роботу
"""
import copy
import os

from core.model import Problem
from core.generator import generate
from core import io_json
from algorithms import greedy, genetic, annealing, exact
from experiments import runner

OUT_DIR = os.path.join("results", "cli")   # куди зберігаються результати з меню

current_problem = None          # задача, з якою зараз працює користувач
ga_params = {                   # поточні параметри генетичного алгоритму
    "I": genetic.DEFAULT_I,
    "k": genetic.DEFAULT_K,
    "p_c": genetic.DEFAULT_PC,
    "p_m": None,                # None означає p_m = 1/n
    "tau": genetic.DEFAULT_TAU,
}
sa_params = {                   # поточні параметри імітації відпалу
    "T0": annealing.DEFAULT_T0,
    "alpha": annealing.DEFAULT_ALPHA,
    "T_min": annealing.DEFAULT_T_MIN,
    "moves": annealing.DEFAULT_MOVES,
}


def out_path(name):
    """Файл у папці results/cli (якщо користувач не вказав іншої папки)."""
    if os.path.dirname(name):
        return name
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


# ===================== введення значень =====================

def ask_int(prompt, lo=None, hi=None, default=None):
    while True:
        text = input(prompt).strip()
        if text == "" and default is not None:
            return default
        try:
            value = int(text)
        except ValueError:
            print("  Помилка: введіть ціле число.")
            continue
        if lo is not None and value < lo:
            print(f"  Значення має бути не менше {lo}.")
        elif hi is not None and value > hi:
            print(f"  Значення має бути не більше {hi}.")
        else:
            return value


def ask_float(prompt, lo=None, default=None):
    while True:
        text = input(prompt).strip().replace(",", ".")
        if text == "" and default is not None:
            return default
        try:
            value = float(text)
        except ValueError:
            print("  Помилка: введіть число.")
            continue
        if lo is not None and value < lo:
            print(f"  Значення має бути не менше {lo}.")
        else:
            return value


def ask_yes_no(prompt):
    while True:
        text = input(prompt).strip().lower()
        if text in ("так", "т", "y", "yes"):
            return True
        if text in ("ні", "н", "n", "no"):
            return False
        print("  Введіть «так» або «ні».")


def ask_int_list(prompt, default):
    """Список цілих чисел через пробіл; Enter — значення за замовчуванням."""
    while True:
        text = input(f"{prompt} [{' '.join(map(str, default))}]: ").strip()
        if text == "":
            return list(default)
        try:
            return [int(x) for x in text.split()]
        except ValueError:
            print("  Помилка: введіть цілі числа через пробіл.")


def ask_range(prompt, default):
    """Діапазон «від до крок»; Enter — список за замовчуванням."""
    while True:
        text = input(f"{prompt} (від до крок) [{' '.join(map(str, default))}]: ").strip()
        if text == "":
            return list(default)
        try:
            start, stop, step = (int(x) for x in text.split())
            return list(range(start, stop + 1, step))
        except ValueError:
            print("  Помилка: введіть три цілі числа, наприклад «50 300 50».")


# ===================== індивідуальна задача =====================

def input_problem_manual():
    print("\n--- Введення даних задачі вручну ---")
    n = ask_int("Кількість модулів n: ", lo=1)
    h = ask_int("Кількість напрямів h: ", lo=1)
    a, b, direction = [], [], []
    for i in range(n):
        print(f"Модуль {i + 1}:")
        a.append(ask_float("  витрати ресурсу a: ", lo=0.01))
        b.append(ask_float("  базовий ефект b: ", lo=0))
        direction.append(ask_int(f"  напрям (1..{h}): ", lo=1, hi=h) - 1)
    A = ask_float("Сумарний доступний ресурс A: ", lo=0.01)
    u, P = [], set()
    for t in range(h):
        u.append(ask_int(f"Квота напряму {t + 1}: ", lo=0))
        if ask_yes_no(f"Напрям {t + 1} пріоритетний? (так/ні): "):
            P.add(t)
    L = ask_int("Мінімум модулів з пріоритетних напрямів L: ", lo=0)
    pairs = {}
    for _ in range(ask_int("Кількість пар взаємопідсилення: ", lo=0)):
        i = ask_int(f"  перший модуль пари (1..{n}): ", lo=1, hi=n) - 1
        j = ask_int(f"  другий модуль пари (1..{n}): ", lo=1, hi=n) - 1
        if i == j:
            print("  Модулі пари мають бути різними — пару пропущено.")
            continue
        pairs[tuple(sorted((i, j)))] = ask_float("  бонус γ: ", lo=0.01)
    return Problem(n=n, h=h, A=A, a=a, b=b, direction=direction,
                   u=u, P=P, L=L, pairs=pairs)


def input_problem_random():
    print("\n--- Генерація задачі (Enter — значення за замовчуванням) ---")
    n = ask_int("Кількість модулів n: ", lo=2)
    h = ask_int("Кількість напрямів h [5]: ", lo=1, default=5)
    rho = ask_float("Жорсткість бюджету ρ, A = ρ·Σa [0.4]: ", lo=0.01, default=0.4)
    mu = ask_float("Щільність пар μ, |W| = μ·n [1.5]: ", lo=0, default=1.5)
    pi = ask_float("Частка пріоритетних напрямів π [0.4]: ", lo=0, default=0.4)
    seed = ask_int("Зерно генерації seed [1]: ", default=1)
    p = generate(n, h=h, rho=rho, mu=mu, pi=pi, seed=seed)
    print(f"  Згенеровано: n={p.n}, h={p.h}, A={p.A}, |W|={len(p.pairs)}, L={p.L}")
    return p


def input_problem_file():
    path = input("Шлях до JSON-файлу: ").strip()
    if not os.path.isfile(path):
        print(f"  Файл «{path}» не знайдено.")
        return None
    try:
        return io_json.load_problem(path)
    except (ValueError, KeyError, TypeError) as error:
        print(f"  Файл має неправильний формат: {error}")
        return None


def edit_problem(p):
    """Редагування окремих параметрів поточної задачі."""
    while True:
        print("\n--- Редагування задачі ---")
        print("  1. Змінити параметри модуля (a, b, напрям)")
        print("  2. Змінити ресурс A")
        print("  3. Змінити квоту напряму")
        print("  4. Змінити мінімум L")
        print("  5. Додати або змінити пару взаємопідсилення")
        print("  0. Готово")
        choice = input("Введіть число: ").strip()
        if choice == "1":
            i = ask_int(f"Номер модуля (1..{p.n}): ", lo=1, hi=p.n) - 1
            p.a[i] = ask_float(f"  a [{p.a[i]}]: ", lo=0.01, default=p.a[i])
            p.b[i] = ask_float(f"  b [{p.b[i]}]: ", lo=0, default=p.b[i])
            p.direction[i] = ask_int(f"  напрям [{p.direction[i] + 1}]: ", lo=1, hi=p.h,
                                     default=p.direction[i] + 1) - 1
        elif choice == "2":
            p.A = ask_float(f"A [{p.A}]: ", lo=0.01, default=p.A)
        elif choice == "3":
            t = ask_int(f"Номер напряму (1..{p.h}): ", lo=1, hi=p.h) - 1
            p.u[t] = ask_int(f"  квота [{p.u[t]}]: ", lo=0, default=p.u[t])
        elif choice == "4":
            p.L = ask_int(f"L [{p.L}]: ", lo=0, default=p.L)
        elif choice == "5":
            i = ask_int(f"Перший модуль (1..{p.n}): ", lo=1, hi=p.n) - 1
            j = ask_int(f"Другий модуль (1..{p.n}): ", lo=1, hi=p.n) - 1
            if i != j:
                p.pairs[tuple(sorted((i, j)))] = ask_float("  бонус γ: ", lo=0.01)
        elif choice == "0":
            p.__post_init__()           # оновити списки сусідів після змін
            return p
        else:
            print("  Невідома команда.")


def menu_problem():
    global current_problem
    while True:
        print("\n--- Робота з індивідуальною задачею ---")
        print("  1. Ввести дані вручну")
        print("  2. Згенерувати дані випадковим чином")
        print("  3. Зчитати дані з файлу")
        print("  4. Редагувати поточну задачу")
        print("  5. Зберегти поточну задачу у файл")
        print("  0. Повернутися в головне меню")
        choice = input("Введіть число: ").strip()
        p = None
        if choice == "1":
            p = input_problem_manual()
        elif choice == "2":
            p = input_problem_random()
        elif choice == "3":
            p = input_problem_file()
        elif choice in ("4", "5") and current_problem is None:
            print("  Задачу ще не задано.")
        elif choice == "4":
            p = edit_problem(copy.deepcopy(current_problem))
        elif choice == "5":
            path = out_path(input("Назва файлу: ").strip())
            io_json.save_problem(current_problem, path)
            print(f"  Задачу збережено у «{path}».")
        elif choice == "0":
            return
        else:
            print("  Невідома команда.")

        if p is not None:
            ok, message = io_json.validate_problem(p)
            if ok:
                current_problem = p
                print("  Задача коректна і стала поточною.")
            else:
                print(f"  Помилка в даних: {message}")


# ===================== розв'язання =====================

def format_solution(S):
    return "{" + ", ".join(str(i + 1) for i in sorted(S)) + "}"


def solve_all():
    if current_problem is None:
        print("\n  Спочатку задайте задачу (пункт 1).")
        return
    p = current_problem
    print("\n--- Розв'язання задачі ---")
    results = [("Жадібний алгоритм", greedy.solve(p)),
               ("Генетичний алгоритм", genetic.solve(p, seed=1, **ga_params)),
               ("Імітація відпалу", annealing.solve(p, seed=1, **sa_params))]
    optimum = exact.brute_force(p)          # None, якщо n завелике
    if optimum is not None:
        results.append(("Точний розв'язок (повний перебір)", optimum))
    F_lp = exact.lp_upper_bound(p)

    lines = []
    for name, sol in results:
        lines.append(f"{name}:")
        lines.append(f"  S = {format_solution(sol.S)}")
        lines.append(f"  F = {sol.F:g};  ресурс {p.cost(sol.S):g} з {p.A:g};  "
                     f"допустимий: {'так' if p.is_feasible(sol.S) else 'НІ'}")
        extra = ""
        if name.startswith("Генетичний"):
            extra = f";  поколінь {sol.generations}"
        elif name.startswith("Імітація"):
            extra = f";  рівнів температури {sol.generations}"
        lines.append(f"  час = {sol.time * 1000:.2f} мс{extra}")
    lines.append(f"Верхня межа (LP-релаксація): F_LP = {F_lp:.2f}")
    reference = optimum.F if optimum is not None else F_lp
    reference_name = "оптимуму" if optimum is not None else "F_LP"
    for name, sol in results[:3]:
        lines.append(f"Відхилення: {name.lower()} від {reference_name} — "
                     f"{runner.deviation(sol.F, reference):.2f} %")
    print("\n".join(lines))

    if ask_yes_no("\nЗберегти результати у файл? (так/ні): "):
        path = out_path(input("Назва файлу: ").strip())
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Результати збережено у «{path}».")


def menu_ga_params():
    print("\n--- Параметри генетичного алгоритму (Enter — залишити) ---")
    ga_params["I"] = ask_int(f"Розмір популяції I [{ga_params['I']}]: ", lo=4,
                             default=ga_params["I"])
    ga_params["k"] = ask_int(f"Поколінь без покращення k [{ga_params['k']}]: ", lo=1,
                             default=ga_params["k"])
    ga_params["p_c"] = ask_float(f"Імовірність схрещення p_c [{ga_params['p_c']}]: ",
                                 lo=0, default=ga_params["p_c"])
    p_m = ask_float("Імовірність мутації p_m (0 — означає 1/n) "
                    f"[{ga_params['p_m'] or 0}]: ", lo=0, default=ga_params["p_m"] or 0)
    ga_params["p_m"] = p_m if p_m > 0 else None
    ga_params["tau"] = ask_int(f"Розмір турніру τ [{ga_params['tau']}]: ", lo=1,
                               default=ga_params["tau"])
    print("--- Параметри імітації відпалу (Enter — залишити) ---")
    sa_params["T0"] = ask_float(f"Початкова температура T0 [{sa_params['T0']}]: ", lo=0.001,
                                default=sa_params["T0"])
    sa_params["alpha"] = ask_float(f"Коефіцієнт охолодження α [{sa_params['alpha']}]: ",
                                   lo=0.01, default=sa_params["alpha"])
    sa_params["T_min"] = ask_float(f"Кінцева температура T_min [{sa_params['T_min']}]: ",
                                   lo=0.0001, default=sa_params["T_min"])
    sa_params["moves"] = ask_int(f"Спроб переходу на температуру [{sa_params['moves']}]: ",
                                 lo=1, default=sa_params["moves"])


# ===================== експерименти =====================

def menu_experiments():
    print("\n--- Проведення експериментів (Enter — значення за замовчуванням) ---")
    print("  1. Визначення параметра умови завершення k (ГА)")
    print("  2. Вплив розміру популяції I (ГА)")
    print("  3. Вплив коефіцієнта охолодження α (ІВ)")
    print("  4. Вплив розмірності задачі на точність і час")
    print(f"  (результати зберігаються в папку {OUT_DIR})")
    print("  0. Повернутися в головне меню")
    choice = input("Введіть число: ").strip()
    if choice == "1":
        n = ask_int("Розмірність задач n [100]: ", lo=2, default=100)
        R = ask_int("Кількість задач R [20]: ", lo=2, default=20)
        k_values = ask_int_list("Значення k", (5, 10, 20, 30, 50, 75, 100, 150))
        runner.experiment_stop(n=n, R=R, k_values=k_values, out_dir=OUT_DIR)
    elif choice == "2":
        n = ask_int("Розмірність задач n [100]: ", lo=2, default=100)
        R = ask_int("Кількість задач R [20]: ", lo=2, default=20)
        I_values = ask_int_list("Значення I", (10, 20, 30, 50, 75, 100, 150))
        runner.experiment_population(n=n, R=R, I_values=I_values, out_dir=OUT_DIR)
    elif choice == "3":
        n = ask_int("Розмірність задач n [100]: ", lo=2, default=100)
        R = ask_int("Кількість задач R [20]: ", lo=2, default=20)
        text = input("Значення α через пробіл [0.80 0.85 0.90 0.95 0.97 0.99]: ").strip()
        alphas = [float(x) for x in text.replace(",", ".").split()] if text else \
            [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
        runner.experiment_cooling(n=n, R=R, alpha_values=alphas, out_dir=OUT_DIR)
    elif choice == "4":
        n_values = ask_range("Розмірності n", (10, 15, 20, 50, 100, 200, 300, 500))
        R = ask_int("Кількість задач на кожну розмірність R [20]: ", lo=2, default=20)
        runner.experiment_size(n_values=n_values, R=R, out_dir=OUT_DIR)
    elif choice != "0":
        print("  Невідома команда.")


# ===================== виведення задачі =====================

def show_problem():
    if current_problem is None:
        print("\n  Задачу ще не задано.")
        return
    p = current_problem
    print("\n--- Дані поточної задачі ---")
    print(f"n = {p.n}, h = {p.h}, A = {p.A:g}, L = {p.L}")
    print(f"Пріоритетні напрями P = {format_solution(p.P)}")
    print(f"Квоти u = {p.u}")
    print(f"{'Модуль':>7} {'Напрям':>7} {'a':>7} {'b':>7}")
    for i in range(p.n):
        print(f"{i + 1:>7} {p.direction[i] + 1:>7} {p.a[i]:>7g} {p.b[i]:>7g}")
    print("Пари взаємопідсилення W:")
    for (i, j), gamma in sorted(p.pairs.items()):
        print(f"  ({i + 1}, {j + 1}): γ = {gamma:g}")


# ===================== головне меню =====================

def main():
    print("=" * 60)
    print(" Планування впровадження модулів у медичній інформаційній")
    print(" системі з квотами за напрямами та ефектами взаємопідсилення")
    print("=" * 60)
    actions = {"1": menu_problem, "2": solve_all, "3": menu_ga_params,
               "4": menu_experiments, "5": show_problem}
    while True:
        status = "задано" if current_problem is not None else "не задано"
        print(f"\n=== ГОЛОВНЕ МЕНЮ ===  (задача: {status})")
        print("  1. Робота з індивідуальною задачею")
        print("  2. Розв'язати задачу всіма алгоритмами")
        print("  3. Параметри наближених алгоритмів (ГА та ІВ)")
        print("  4. Провести експерименти")
        print("  5. Вивести дані задачі")
        print("  0. Завершити роботу")
        choice = input("Введіть число: ").strip()
        if choice == "0":
            print("Роботу програми завершено.")
            break
        if choice in actions:
            actions[choice]()
        else:
            print("  Невідома команда, спробуйте ще раз.")


if __name__ == "__main__":
    main()
