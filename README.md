# Corners RL — DQN Self-Play in the Game of «Уголки»

Исследовательский проект: обучение DQN-агента в игре «Уголки» методом self-play с
comparison двух стратегий experience replay — **Uniform Replay** и **Prioritized Experience
Replay (PER)**. Содержит полную среду, три baseline-агента, воспроизводимый
evaluation pipeline с bootstrap confidence intervals и отчёт в формате LaTeX.

---

## Что реализовано

| Компонент | Описание |
|---|---|
| **Среда** | Ортогональные «Уголки» 8×8, 9 фишек у каждого; прыжки + цепочки прыжков |
| **Baseline-агенты** | Random, Greedy (Манхэттен), Heuristic (многофакторный) |
| **DQN** | CNN 3×8×8 → 4096 Q-values; legal action mask; ε-greedy self-play |
| **Reward shaping** | Победа ±100; прогресс к цели; штраф за медлительность |
| **Imitation warm-start** | Behavioural Cloning от HeuristicAgent перед self-play |
| **Replay ablation** | Сравнение Uniform Replay и PER (α=0.6, β 0.4→1.0) |
| **Bootstrap CI** | 95% CI по отдельным партиям, разностный CI для сравнения методов |
| **Отчёт** | LaTeX-тезисы в `report/Шаблон_тезисов.tex` |

---

## Структура репозитория

```
UGOLKI/
├── configs/
│   ├── dqn.yaml               # Базовые гиперпараметры
│   ├── dqn_uniform.yaml       # Конфиг Uniform Replay
│   └── dqn_per.yaml           # Конфиг PER
├── scripts/
│   ├── train_dqn.py           # Обучение одного агента
│   ├── run_replay_ablation.py # Полный ablation: Uniform × seeds + PER × seeds
│   ├── evaluate_agents.py     # Турнир агентов → CSV
│   ├── eval_bootstrap.py      # Крупный eval + 95% bootstrap CI (все seeds)
│   ├── eval_draw_cutoff.py    # Ablation по max_moves (draw cutoff)
│   ├── eval_all_seeds.py      # Per-seed variance + boxplot
│   ├── plot_replay_ablation.py# Графики ablation-эксперимента
│   ├── pretrain_imitation.py  # Behavioural Cloning от HeuristicAgent
│   └── visualize_game.py      # GIF-анимация партии
├── src/corners_rl/
│   ├── agents/                # RandomAgent, GreedyAgent, HeuristicAgent, DQNAgent
│   ├── env/                   # CornersEnv, moves.py, rules.py
│   ├── evaluation/            # evaluate_match, round_robin_tournament
│   ├── rl/                    # DQNModel, ReplayBuffer, SelfPlayTrainer, TrainConfig
│   └── visualization/         # board_plot, animate_game, plots
├── tests/                     # pytest-тесты (env, agents, encoding, training)
├── report/
│   ├── Шаблон_тезисов.tex     # LaTeX-отчёт с актуальными результатами
│   └── figures/               # PNG-графики для отчёта
└── outputs/
    ├── models/                # imitation.pt (warm-start checkpoint)
    ├── experiments/main/      # Основной ablation: seeds 1-2
    ├── experiments/extra_seeds/ # Дополнительные seeds 3-5
    ├── eval_bootstrap/        # Bootstrap eval (seeds 1-2, 1000 игр/пару)
    ├── eval_draw_cutoff/      # Ablation по draw cutoff
    └── eval_all_seeds/        # Per-seed eval (все seeds)
```

---

## Правила игры

Ортогональная версия «Уголков»: доска 8×8, у каждого игрока 9 фишек.

- **Шаг** — на одну свободную клетку по вертикали или горизонтали.
- **Прыжок** — через занятую клетку на 2 позиции ортогонально; цепочки прыжков разрешены.
- **Победа** — первый, кто переместил все 9 фишек в противоположный угол.
- **Ничья** — если достигнут лимит `max_moves` ходов.

Диагональные ходы **запрещены**. Каждый ход валидируется `validate_move()`.

---

## Метод

### Кодирование состояния

Тензор 3×8×8:

| Канал | Содержание |
|---|---|
| 0 | Свои фишки |
| 1 | Фишки соперника |
| 2 | Целевая зона |

Для Player −1 доска поворачивается на 180° — обе стороны видят задачу в единой
системе координат. Это позволяет использовать **одну сеть за обоих игроков**.

### Пространство действий

4096 действий (64×64 = from×to). Legal action mask маскирует нелегальные ходы в −∞
до `argmax`, поэтому агент физически не может выбрать нелегальный ход.

### Архитектура DQN

```
Input: 3×8×8
→ Conv2d(3→32, k=3, pad=1) + ReLU
→ Conv2d(32→64, k=3, pad=1) + ReLU
→ Flatten → Linear(4096→512) → ReLU → Linear(512→4096)
Output: Q-values [4096]
```

### Self-Play и Negamax bootstrapping

В self-play `next_state` кодируется с точки зрения **соперника**. В zero-sum игре
`V(s, игрок) = −V(s, соперник)`, поэтому target Беллмана использует **вычитание**:

```
target = reward − γ · max_Q(next_state) · (1 − done)
```

Знак «+» вместо «−» — классическая ошибка, ведущая к 0% win rate.

### PER (Prioritized Experience Replay)

```
p_i = (|δ_i| + ε)^α,   P(i) = p_i / Σ p_j
```

IS-веса корректируют смещение: β анилингуется от 0.4 до 1.0. α=0.6.

---

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Для Apple Silicon (MPS) достаточно стандартного `torch>=2.0`.

---

## Быстрый старт

```bash
# Проверка правил (0 диагональных ходов)
PYTHONPATH=src python scripts/check_rules.py --games 20 --seed 42

# Тестовое обучение (5 эпизодов, CPU)
PYTHONPATH=src python scripts/train_dqn.py --episodes 5 --device cpu --seed 42

# Базовый турнир без DQN
PYTHONPATH=src python scripts/evaluate_agents.py --games 50 --device cpu --seed 42
```

---

## Воспроизведение основного эксперимента

```bash
# 1. Imitation warm-start (опционально, но рекомендуется)
PYTHONPATH=src python scripts/pretrain_imitation.py \
    --games 300 --epochs 5 \
    --out outputs/models/imitation.pt

# 2. Ablation: Uniform vs PER, seeds 1-2, 1500 эпизодов (~3 ч на MPS)
PYTHONPATH=src python scripts/run_replay_ablation.py \
    --episodes 1500 --seeds 1 2 --device auto \
    --max-moves 300 --eval-games 100 \
    --out outputs/experiments/main

# 3. Дополнительные seeds 3-5 (то же конфиг)
PYTHONPATH=src python scripts/run_replay_ablation.py \
    --episodes 1500 --seeds 3 4 5 --device auto \
    --max-moves 300 --eval-games 50 \
    --out outputs/experiments/extra_seeds

# 4. Bootstrap evaluation (1000 игр/пару, seeds 1-2)
PYTHONPATH=src python scripts/eval_bootstrap.py \
    --games 1000 --device auto \
    --out outputs/eval_bootstrap

# 5. Per-seed variance (все available seeds)
PYTHONPATH=src python scripts/eval_all_seeds.py \
    --games 500 --device auto \
    --out outputs/eval_all_seeds

# 6. Draw cutoff ablation (max_moves 300/450/600)
PYTHONPATH=src python scripts/eval_draw_cutoff.py \
    --max-moves-list 300 450 600 --games 1000 --device auto \
    --out outputs/eval_draw_cutoff

# 7. Генерация графиков ablation
PYTHONPATH=src python scripts/plot_replay_ablation.py \
    --experiment-dir outputs/experiments/main \
    --out outputs/experiments/main/figures
```

---

## Результаты (bootstrap, 1000 игр/пару)

| Стратегия | Win rate | 95% CI |
|---|---|---|
| Uniform Replay | 23.5% | [22.4%, 24.6%] |
| Prioritized ER  | 17.4% | [16.4%, 18.3%] |

Разностный CI (Uniform − PER): **+6.1% [+4.6%, +7.5%]** — нуль не входит.
Оба агента значимо превосходят Random (0%): нижняя граница CI всех seed ≥ 12%.

Результаты могут измениться при увеличении числа seeds (см. `eval_all_seeds.py`).

---

## Тесты

```bash
pytest -q
```

Покрыты: легальность ходов DQN, атрибуция победителя в evaluation,
изменение весов после `dqn_update`, reward shaping, кодирование для Player −1.

---

## Ограничения

- Цепочки прыжков с одинаковыми start/end клетками получают одинаковый `action_id`.
- Draw cutoff `max_moves=300` влияет на draw rate; sensitivity проверена ablation-ом.
- При малом числе seeds (n=2) difference CI может включать 0 — нужно ≥ 4–5 seeds для robust claim.
