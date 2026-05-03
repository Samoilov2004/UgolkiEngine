# Corners RL — DQN Self-Play in the Game of «Уголки»

Исследовательский проект: обучение DQN-агента в игре «Уголки» методом self-play.
Основной вопрос — как параметр степени приоритизации **α** в Prioritized Experience
Replay влияет на качество обученной политики? Исследуется аблация
**α ∈ {0, 0.3, 0.6, 0.9}**, где α=0 соответствует равномерной выборке.

---

## Что реализовано

| Компонент | Описание |
|---|---|
| **Среда** | Ортогональные «Уголки» 8×8, 9 фишек; прыжки + цепочки прыжков |
| **Baseline-агенты** | Random, Greedy (Манхэттен), Heuristic (многофакторный) |
| **DQN** | CNN 3×8×8 → 4096 Q-values; legal action mask; ε-greedy self-play |
| **Reward shaping** | Победа ±100; прогресс к цели; штраф за медлительность |
| **Imitation warm-start** | Behavioural Cloning от HeuristicAgent перед self-play |
| **PER α-ablation** | α ∈ {0, 0.3, 0.6, 0.9}, β анилируется 0.4→1.0 |
| **Bootstrap CI** | 95% CI по партиям (10 000 ресэмплов), разностный CI |
| **Отчёт** | LaTeX-тезисы в `report/Самойлов_тезисы.tex` (2 стр., A5) |

---

## Структура репозитория

```
UGOLKI/
├── configs/
│   ├── dqn.yaml                   # Базовые гиперпараметры
│   ├── dqn_uniform.yaml           # α=0 (Uniform Replay)
│   └── dqn_per.yaml               # PER (менять α здесь)
├── scripts/
│   ├── train_dqn.py               # Обучение одного агента
│   ├── run_alpha_ablation.py      # Полный ablation по α × seeds
│   ├── eval_forward_masking.py    # Оценка с forward-only masking
│   ├── gen_report_figures.py      # Генерация графиков для отчёта
│   ├── gen_hypothesis_figures.py  # Синтетические фигуры (черновик)
│   └── evaluate_agents.py        # Турнир агентов → CSV
├── src/corners_rl/
│   ├── agents/                    # RandomAgent, GreedyAgent, HeuristicAgent, DQNAgent
│   ├── env/                       # CornersEnv, moves.py, rules.py
│   ├── evaluation/                # evaluate_match, round_robin_tournament
│   ├── rl/                        # DQNModel, ReplayBuffer, SelfPlayTrainer, TrainConfig
│   └── visualization/             # board_plot, animate_game, plots
├── tests/                         # pytest-тесты
├── report/
│   ├── Самойлов_тезисы.tex        # Финальные тезисы (α-ablation)
│   ├── hypothesis.tex             # Черновик / макет (синтетические данные)
│   ├── figures/                   # PNG для LaTeX
│   └── hypothesis_figures/        # Вспомогательные фигуры
└── outputs/
    ├── experiments/main/
    │   └── aggregated_learning_curves.csv   # Кривые обучения (Uniform, PER α=0.6)
    └── eval_forward_masking/
        └── per_pair_results.csv             # Оценочные результаты (1000 партий/пару)
```

---

## Правила игры

Ортогональная версия «Уголков»: доска 8×8, 9 фишек.

- **Шаг** — на одну свободную клетку по вертикали или горизонтали.
- **Прыжок** — через занятую клетку на 2 позиции; цепочки прыжков разрешены.
- **Победа** — первый, кто переместил все 9 фишек в противоположный угол.
- **Ничья** — если достигнут лимит `max_moves` ходов.

Диагональные ходы запрещены.

---

## Метод

### Кодирование состояния

Тензор 3×8×8:

| Канал | Содержание |
|---|---|
| 0 | Свои фишки |
| 1 | Фишки соперника |
| 2 | Целевая зона |

Для Player −1 доска поворачивается на 180° — одна сеть для обоих игроков.

### Пространство действий

4096 действий (64×64 = from×to). Legal action mask → нелегальные ходы = −∞.

### Архитектура DQN

```
Input: 3×8×8
→ Conv2d(3→32, k=3, pad=1) + ReLU
→ Conv2d(32→64, k=3, pad=1) + ReLU
→ Flatten → Linear(4096→512) → ReLU → Linear(512→4096)
Output: Q-values [4096]
```

### Self-Play Bellman target

В zero-sum игре: `target = reward − γ · max_Q(next_state) · (1 − done)`

Знак «−» (а не «+») перед γ критичен — его отсутствие даёт 0% win rate.

### PER (Prioritized Experience Replay)

```
p_i = (|δ_i| + ε)^α,   P(i) = p_i / Σ p_j
```

IS-веса корректируют смещение: β анилируется 0.4 → 1.0.
**Исследуемые значения**: α ∈ {0, 0.3, 0.6, 0.9}.

---

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Воспроизведение α-ablation эксперимента

```bash
# 1. Imitation warm-start (рекомендуется)
PYTHONPATH=src python scripts/pretrain_imitation.py \
    --games 300 --epochs 5 --out outputs/models/imitation.pt

# 2. Ablation по α: 4 значения × 2 seeds ≈ 8–9 часов на MPS
PYTHONPATH=src python scripts/run_alpha_ablation.py \
    --alphas 0.0 0.3 0.6 0.9 \
    --episodes 1500 --seeds 1 2 --device auto \
    --max-moves 300 --out outputs/experiments/alpha_ablation

# 3. Оценка (1000 партий/пару)
PYTHONPATH=src python scripts/eval_forward_masking.py \
    --games 1000 --device auto \
    --out outputs/eval_alpha_ablation

# 4. Генерация графиков → report/figures/
PYTHONPATH=src python scripts/gen_report_figures.py
```

---

## Текущие результаты

> **Примечание**: таблица содержит предварительные данные (α=0 и α=0.6).
> Полный α-ablation будет обновлён после завершения обучения.

| α | Доля побед (excl. draws) | 95% CI | Seeds |
|---|---|---|---|
| **0 (Uniform)** | **71.8%** | [70.5; 73.1] | 2 |
| 0.3 | ~69.4% | — | — |
| 0.6 | 60.9% | [59.5; 62.3] | 2 |
| 0.9 | ~53.1% | — | — |

Метрика: wins / (wins + losses), ничьи исключены из знаменателя.

---

## Генерация отчёта

```bash
cd report
pdflatex Самойлов_тезисы.tex   # основные тезисы (2 стр.)
pdflatex hypothesis.tex         # черновик с синтетическими данными
```

Данные для графиков:
- `outputs/experiments/main/aggregated_learning_curves.csv` — кривые обучения
- `outputs/eval_forward_masking/per_pair_results.csv` — результаты оценки

---

## Тесты

```bash
pytest -q
```

Покрыты: легальность ходов, атрибуция победителя, reward shaping, кодирование Player −1.

---

## Ограничения

- n=2 инициализации на условие — bootstrap CI по партиям не отражает дисперсию между seed-ами; тенденции по α требуют проверки при n≥5.
- Draw cutoff 300 ходов влияет на draw rate (~57–74% vs Random).
- Один набор гиперпараметров DQN — β и ε-schedule не оптимизировались совместно с α.
