"""
experiments/runner.py — експериментальне дослідження алгоритмів (розділи 3 і 5 КР).

Чотири експерименти:
  1) experiment_stop       — вибір параметра k умови завершення ГА;
  2) experiment_population — вплив розміру популяції I на роботу ГА;
  3) experiment_cooling    — вплив коефіцієнта охолодження α на роботу ІВ;
  4) experiment_size       — вплив розмірності n на точність і час
                             жадібного алгоритму (ЖА), ГА та ІВ.

Точність оцінюємо відносним відхиленням від верхньої межі F_LP:
    δ = (F_LP - F) / F_LP * 100 %      (чим менше, тим краще),
а для малих задач — ще й від точного оптимуму F* (повний перебір).

Кожен експеримент повторюється на R різних задачах (R — кількість
прогонів), результати усереднюються і зберігаються в папку out_dir
у вигляді таблиці CSV та графіка PNG.

Запуск усіх експериментів:  python -m experiments.runner
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")           # малюємо графіки у файл, без вікна
import matplotlib.pyplot as plt

from core.generator import generate
from algorithms import greedy, genetic, annealing, exact

H = 5                   # кількість напрямів у згенерованих задачах
SEED = 1                # зерно ГА та ІВ (однакове — для чесного порівняння)
EXACT_MAX_N = 15        # до якої розмірності рахуємо точний оптимум
RESULTS_DIR = "results"


def deviation(F, F_best):
    """Відносне відхилення F від еталонного значення F_best, %."""
    return (F_best - F) / F_best * 100


def mean(values):
    return statistics.mean(values)


def save_csv(out_dir, name, header, rows):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Таблицю збережено: {path}")


def save_plot(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Графік збережено: {path}")


def plot_two_axes(x, y_dev, y_time, xlabel, title, out_dir, name):
    """Графік: ліва вісь — відхилення, права — час."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, y_dev, "o-", color="#1f77b4", lw=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Середнє відхилення від F_LP, %", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(x, y_time, "s--", color="#d62728", lw=2)
    ax2.set_ylabel("Середній час роботи, с", color="#d62728")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    save_plot(fig, out_dir, name)


# ======================================================================
# Експеримент 1. Параметр умови завершення k (ГА)
# ======================================================================

def experiment_stop(n=100, R=20, k_values=(5, 10, 20, 30, 50, 75, 100, 150),
                    dynamics_generations=300, out_dir=RESULTS_DIR):
    print(f"\nЕксперимент 1: вибір параметра k (n={n}, R={R})")
    problems = [generate(n, h=H, seed=r) for r in range(R)]
    bounds = [exact.lp_upper_bound(p) for p in problems]

    # --- 1а. Динаміка рекорду: ГА без ранньої зупинки ---
    histories, last_improvements = [], []
    for p, F_lp in zip(problems, bounds):
        sol = genetic.solve(p, I=50, k=dynamics_generations,
                            max_generations=dynamics_generations, seed=SEED)
        histories.append([deviation(F, F_lp) for F in sol.history])
        # номер покоління, на якому рекорд покращився востаннє
        last = max(g for g in range(len(sol.history))
                   if g == 0 or sol.history[g] > sol.history[g - 1])
        last_improvements.append(last)
    avg_history = [mean(h[g] for h in histories)
                   for g in range(dynamics_generations + 1)]
    print(f"  Останнє покращення рекорду: в середньому на поколінні "
          f"{mean(last_improvements):.1f}, найпізніше — {max(last_improvements)}")

    save_csv(out_dir, "exp_stop_dynamics.csv", ["generation", "avg_deviation_LP_%"],
             [[g, f"{d:.3f}"] for g, d in enumerate(avg_history)])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(dynamics_generations + 1), avg_history, color="#1f77b4", lw=2)
    ax.set_xlabel("Номер покоління")
    ax.set_ylabel("Середнє відхилення рекорду від F_LP, %")
    ax.set_title(f"Динаміка рекорду ГА (n = {n}, R = {R})")
    ax.grid(alpha=0.3)
    save_plot(fig, out_dir, "exp_stop_dynamics.png")

    # --- 1б. Розв'язання тих самих задач з різними k ---
    rows = []
    for k in k_values:
        devs, times, gens = [], [], []
        for p, F_lp in zip(problems, bounds):
            sol = genetic.solve(p, I=50, k=k, seed=SEED)
            devs.append(deviation(sol.F, F_lp))
            times.append(sol.time)
            gens.append(sol.generations)
        rows.append([k, mean(devs), statistics.stdev(devs), mean(times), mean(gens)])
        print(f"  k={k:4d}: відхилення {mean(devs):.3f} %, "
              f"час {mean(times):.3f} с, поколінь {mean(gens):.0f}")

    save_csv(out_dir, "exp_stop.csv",
             ["k", "avg_deviation_LP_%", "stdev_%", "avg_time_s", "avg_generations"],
             [[r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.1f}"]
              for r in rows])
    plot_two_axes(k_values, [r[1] for r in rows], [r[3] for r in rows],
                  "Кількість поколінь без покращення k",
                  f"Вплив параметра k на точність і час ГА (n = {n})",
                  out_dir, "exp_stop.png")
    return rows


# ======================================================================
# Експеримент 2. Розмір популяції I (ГА)
# ======================================================================

def experiment_population(n=100, R=20, I_values=(10, 20, 30, 50, 75, 100, 150),
                          k=genetic.DEFAULT_K, out_dir=RESULTS_DIR):
    print(f"\nЕксперимент 2: вплив розміру популяції I (n={n}, R={R}, k={k})")
    problems = [generate(n, h=H, seed=1000 + r) for r in range(R)]
    bounds = [exact.lp_upper_bound(p) for p in problems]

    rows = []
    for I in I_values:
        devs, times = [], []
        for p, F_lp in zip(problems, bounds):
            sol = genetic.solve(p, I=I, k=k, seed=SEED)
            devs.append(deviation(sol.F, F_lp))
            times.append(sol.time)
        rows.append([I, mean(devs), statistics.stdev(devs), mean(times)])
        print(f"  I={I:4d}: відхилення {mean(devs):.3f} %, час {mean(times):.3f} с")

    save_csv(out_dir, "exp_population.csv",
             ["I", "avg_deviation_LP_%", "stdev_%", "avg_time_s"],
             [[r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.4f}"] for r in rows])
    plot_two_axes(I_values, [r[1] for r in rows], [r[3] for r in rows],
                  "Розмір популяції I",
                  f"Вплив розміру популяції на точність і час ГА (n = {n})",
                  out_dir, "exp_population.png")
    return rows


# ======================================================================
# Експеримент 3. Коефіцієнт охолодження α (ІВ)
# ======================================================================

def experiment_cooling(n=100, R=20, alpha_values=(0.80, 0.85, 0.90, 0.95, 0.97, 0.99),
                       out_dir=RESULTS_DIR):
    print(f"\nЕксперимент 3: вплив коефіцієнта охолодження α (n={n}, R={R})")
    problems = [generate(n, h=H, seed=1000 + r) for r in range(R)]
    bounds = [exact.lp_upper_bound(p) for p in problems]

    rows = []
    for alpha in alpha_values:
        devs, times, levels = [], [], []
        for p, F_lp in zip(problems, bounds):
            sol = annealing.solve(p, alpha=alpha, seed=SEED)
            devs.append(deviation(sol.F, F_lp))
            times.append(sol.time)
            levels.append(sol.generations)
        rows.append([alpha, mean(devs), statistics.stdev(devs), mean(times), mean(levels)])
        print(f"  α={alpha:.2f}: відхилення {mean(devs):.3f} %, час {mean(times):.3f} с, "
              f"рівнів температури {mean(levels):.0f}")

    save_csv(out_dir, "exp_cooling.csv",
             ["alpha", "avg_deviation_LP_%", "stdev_%", "avg_time_s", "temperature_levels"],
             [[f"{r[0]:.2f}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.0f}"]
              for r in rows])
    plot_two_axes(alpha_values, [r[1] for r in rows], [r[3] for r in rows],
                  "Коефіцієнт охолодження α",
                  f"Вплив коефіцієнта охолодження на точність і час ІВ (n = {n})",
                  out_dir, "exp_cooling.png")
    return rows


# ======================================================================
# Експеримент 4. Вплив розмірності n на точність і час
# ======================================================================

def experiment_size(n_values=(10, 15, 20, 50, 100, 200, 300, 500), R=20,
                    out_dir=RESULTS_DIR):
    print(f"\nЕксперимент 4: вплив розмірності n (R={R})")
    names = ["ЖА", "ГА", "ІВ"]
    rows = []
    for n in n_values:
        dev = {a: [] for a in names}
        dev_opt = {a: [] for a in names}
        times = {a: [] for a in names}
        ga_wins = sa_wins = 0
        for r in range(R):
            p = generate(n, h=H, seed=2000 + r)
            F_lp = exact.lp_upper_bound(p)
            F_opt = exact.brute_force(p).F if n <= EXACT_MAX_N else None
            sols = {"ЖА": greedy.solve(p),
                    "ГА": genetic.solve(p, seed=SEED),
                    "ІВ": annealing.solve(p, seed=SEED)}
            for a in names:
                dev[a].append(deviation(sols[a].F, F_lp))
                times[a].append(sols[a].time)
                if F_opt is not None:
                    dev_opt[a].append(deviation(sols[a].F, F_opt))
            ga_wins += sols["ГА"].F > sols["ЖА"].F
            sa_wins += sols["ІВ"].F > sols["ЖА"].F
        row = {"n": n, "win_ga": ga_wins / R * 100, "win_sa": sa_wins / R * 100}
        for a in names:
            row["dev_" + a] = mean(dev[a])
            row["opt_" + a] = mean(dev_opt[a]) if dev_opt[a] else None
            row["time_" + a] = mean(times[a])
        rows.append(row)
        print(f"  n={n:4d}: відхилення ЖА {row['dev_ЖА']:.3f} %, ГА {row['dev_ГА']:.3f} %, "
              f"ІВ {row['dev_ІВ']:.3f} %; час ЖА {row['time_ЖА']:.5f} с, "
              f"ГА {row['time_ГА']:.3f} с, ІВ {row['time_ІВ']:.3f} с")

    def fmt(x, digits):
        return "" if x is None else f"{x:.{digits}f}"

    save_csv(out_dir, "exp_size.csv",
             ["n", "greedy_dev_LP_%", "ga_dev_LP_%", "sa_dev_LP_%",
              "greedy_dev_opt_%", "ga_dev_opt_%", "sa_dev_opt_%",
              "ga_win_rate_%", "sa_win_rate_%", "greedy_time_s", "ga_time_s", "sa_time_s"],
             [[r["n"], fmt(r["dev_ЖА"], 3), fmt(r["dev_ГА"], 3), fmt(r["dev_ІВ"], 3),
               fmt(r["opt_ЖА"], 3), fmt(r["opt_ГА"], 3), fmt(r["opt_ІВ"], 3),
               fmt(r["win_ga"], 0), fmt(r["win_sa"], 0),
               fmt(r["time_ЖА"], 6), fmt(r["time_ГА"], 4), fmt(r["time_ІВ"], 4)]
              for r in rows])

    ns = [r["n"] for r in rows]
    style = {"ЖА": ("o-", "#2ca02c", "Жадібний алгоритм"),
             "ГА": ("s-", "#d62728", "Генетичний алгоритм"),
             "ІВ": ("^-", "#9467bd", "Імітація відпалу")}
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in names:
        marker, color, label = style[a]
        ax.plot(ns, [r["dev_" + a] for r in rows], marker, color=color, lw=2, label=label)
    ax.set_xlabel("Кількість модулів n")
    ax.set_ylabel("Середнє відхилення від F_LP, %")
    ax.set_title("Точність алгоритмів залежно від розмірності задачі")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, out_dir, "exp_size_accuracy.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ns, [r["time_ЖА"] * 1000 for r in rows], "o-", color="#2ca02c", lw=2)
    axes[0].set_title("Жадібний алгоритм")
    axes[0].set_ylabel("Середній час роботи, мс")
    for a in ("ГА", "ІВ"):
        marker, color, label = style[a]
        axes[1].plot(ns, [r["time_" + a] for r in rows], marker, color=color, lw=2, label=label)
    axes[1].set_title("Наближені алгоритми")
    axes[1].set_ylabel("Середній час роботи, с")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Кількість модулів n")
        ax.grid(alpha=0.3)
    save_plot(fig, out_dir, "exp_size_time.png")
    return rows


if __name__ == "__main__":
    experiment_stop()
    experiment_population()
    experiment_cooling()
    experiment_size()
