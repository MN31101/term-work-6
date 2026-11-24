"""
model.py — модель задачі планування впровадження модулів МІС.

Тут описано:
  Problem  — дані індивідуальної задачі (позначення як у розділі 1 КР);
  Solution — результат роботи алгоритму.

Розв'язок задачі — це множина S номерів обраних модулів
(те саме, що вектор x: x_i = 1, якщо i ∈ S).
Усередині програми модулі нумеруються з 0, а користувачу
показуються з 1.
"""
from dataclasses import dataclass, field


@dataclass
class Problem:
    n: int              # кількість модулів-кандидатів
    h: int              # кількість функціональних напрямів
    A: float            # сумарний доступний ресурс
    a: list             # a[i] — витрати ресурсу на модуль i
    b: list             # b[i] — базовий ефект модуля i
    direction: list     # direction[i] — номер напряму модуля i (0..h-1)
    u: list             # u[t] — квота напряму t
    P: set              # множина пріоритетних напрямів
    L: int              # мінімум модулів з пріоритетних напрямів
    pairs: dict = field(default_factory=dict)  # {(i, j): γ_ij}, де i < j

    def __post_init__(self):
        # Для кожного модуля запам'ятовуємо його «сусідів» по парах W:
        # neighbors[i] = [(j, γ_ij), ...]. Це пришвидшує обчислення приросту.
        self.neighbors = [[] for _ in range(self.n)]
        for (i, j), gamma in self.pairs.items():
            self.neighbors[i].append((j, gamma))
            self.neighbors[j].append((i, gamma))

    # ---------- допоміжні величини ----------

    def is_priority(self, i):
        """Чи належить модуль i до пріоритетного напряму."""
        return self.direction[i] in self.P

    def cost(self, S):
        """Сумарні витрати ресурсу на модулі з S."""
        return sum(self.a[i] for i in S)

    def count_by_direction(self, S):
        """Скільки модулів з S обрано в кожному напрямі."""
        count = [0] * self.h
        for i in S:
            count[self.direction[i]] += 1
        return count

    def priority_count(self, S):
        """Скільки модулів з S належить пріоритетним напрямам."""
        return sum(1 for i in S if self.is_priority(i))

    # ---------- обмеження та цільова функція ----------

    def is_feasible(self, S):
        """Перевіряє обмеження (1.2)–(1.4) математичної моделі."""
        if self.cost(S) > self.A:                       # ресурс (1.2)
            return False
        count = self.count_by_direction(S)
        for t in range(self.h):                         # квоти (1.3)
            if count[t] > self.u[t]:
                return False
        if self.priority_count(S) < self.L:             # пріоритетні (1.4)
            return False
        return True

    def objective(self, S):
        """Цільова функція (1.6): базові ефекти + бонуси за пари."""
        F = sum(self.b[i] for i in S)
        for (i, j), gamma in self.pairs.items():
            if i in S and j in S:
                F += gamma
        return F

    def has_feasible_solution(self):
        """Чи існує хоча б один допустимий розв'язок.

        Достатньо перевірити, що можна набрати L пріоритетних модулів:
        беремо найдешевші пріоритетні модулі з урахуванням квот.
        Якщо навіть найдешевший такий набір не вміщується в ресурс A,
        то допустимих розв'язків немає."""
        cheapest_first = sorted(
            [i for i in range(self.n) if self.is_priority(i)],
            key=lambda i: self.a[i])
        chosen = set()
        count = [0] * self.h
        for i in cheapest_first:
            if len(chosen) == self.L:
                break
            t = self.direction[i]
            if count[t] < self.u[t]:
                chosen.add(i)
                count[t] += 1
        return len(chosen) == self.L and self.cost(chosen) <= self.A

    def gain(self, S, i):
        """Приріст ЦФ, якщо додати модуль i до S (формула 2.1):
        Δ_i(S) = b_i + сума γ_ij для всіх j ∈ S, що утворюють пару з i."""
        delta = self.b[i]
        for j, gamma in self.neighbors[i]:
            if j in S:
                delta += gamma
        return delta


@dataclass
class Solution:
    S: set                  # обрані модулі
    F: float                # значення цільової функції
    time: float = 0.0       # час роботи алгоритму, с
    generations: int = 0    # кількість поколінь (лише для ГА)
    history: list = field(default_factory=list)  # рекорд F після кожного покоління (ГА)
