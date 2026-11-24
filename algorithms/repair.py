"""
repair.py — процедура ремонту (реанімації) розв'язку для генетичного
алгоритму.

Після схрещення та мутації нащадок може порушувати обмеження.
Ремонт робить його допустимим, змінюючи мінімально:
  крок 1 — якщо в напрямі обрано більше за квоту u_t, вилучаємо
           модулі з найменшим внеском у ЦФ;
  крок 2 — якщо перевищено ресурс A, вилучаємо модулі з найменшим
           внеском на одиницю ресурсу;
  крок 3 — якщо пріоритетних модулів менше за L, додаємо пріоритетні
           модулі з найбільшим приростом на одиницю ресурсу.
Якщо зробити розв'язок допустимим не вдалося — повертаємо None.
"""


def contribution(p, S, i):
    """Внесок модуля i ∈ S у ЦФ: на скільки зменшиться F, якщо його вилучити."""
    return p.gain(S - {i}, i)


def repair(p, S):
    S = set(S)

    # Крок 1. Квоти напрямів (обмеження 1.3)
    for t in range(p.h):
        members = [i for i in S if p.direction[i] == t]
        while len(members) > p.u[t]:
            worst = min(members, key=lambda i: contribution(p, S, i))
            S.remove(worst)
            members.remove(worst)

    # Крок 2. Ресурс (обмеження 1.2)
    used = p.cost(S)
    while used > p.A:
        worst = min(S, key=lambda i: contribution(p, S, i) / p.a[i])
        S.remove(worst)
        used -= p.a[worst]

    # Крок 3. Мінімум пріоритетних модулів (обмеження 1.4)
    count = p.count_by_direction(S)
    while p.priority_count(S) < p.L:
        candidates = [i for i in range(p.n)
                      if i not in S
                      and p.is_priority(i)
                      and used + p.a[i] <= p.A
                      and count[p.direction[i]] < p.u[p.direction[i]]]
        if not candidates:
            return None     # ремонт не вдався
        best = max(candidates, key=lambda i: p.gain(S, i) / p.a[i])
        S.add(best)
        used += p.a[best]
        count[p.direction[best]] += 1

    return S
