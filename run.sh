#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  run.sh — один скрипт для запуска обучения DQN на сервере
#
#  Использование:
#    chmod +x run.sh
#    ./run.sh
#
#  Что произойдёт:
#    1. Создаст Python venv и установит зависимости (один раз ~3 мин)
#    2. Определит число CPU и запустит параллельное обучение на всех ядрах
#    3. После завершения проведёт финальный турнир
#    4. Выведет итоговую таблицу результатов
#    5. Лучший checkpoint сохранится в outputs/models/dqn_best.pt
#
#  Требования: Python 3.10+, интернет для первой установки зависимостей
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── ПАРАМЕТРЫ ───────────────────────────────────────────────────────────────

EPISODES=5000      # эпизодов на каждый seed (больше → лучше, но дольше)
MAX_MOVES=300      # максимум ходов в партии (draw после этого)
EVAL_GAMES=100     # партий на пару в финальном турнире
SEEDS=(1 2 3 4)    # random seeds = кол-во параллельных процессов

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

T_START=$(date +%s)

# ── Цвета ────────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'
    C_CYAN='\033[0;36m';  C_RED='\033[0;31m'; C_RESET='\033[0m'
else
    C_GREEN=''; C_YELLOW=''; C_CYAN=''; C_RED=''; C_RESET=''
fi

log()  { echo -e "${C_CYAN}[$(date '+%H:%M:%S')]${C_RESET}  $*"; }
ok()   { echo -e "${C_GREEN}  ✓${C_RESET}  $*"; }
warn() { echo -e "${C_YELLOW}  ⚠${C_RESET}  $*"; }
fail() { echo -e "${C_RED}  ✗${C_RESET}  $*"; }
sep()  { echo; echo "────────────────────────────────────────────────────────"; }
sep2() { echo; echo "════════════════════════════════════════════════════════"; }

# ─── 1. Python окружение ─────────────────────────────────────────────────────
setup_env() {
    sep2; log "Шаг 1 / 4: Python окружение"

    # Найти python
    PYTHON_BIN=$(command -v python3.12 2>/dev/null || \
                 command -v python3.11 2>/dev/null || \
                 command -v python3.10 2>/dev/null || \
                 command -v python3    2>/dev/null || \
                 echo "")
    if [ -z "$PYTHON_BIN" ]; then
        fail "Python 3.10+ не найден. Установи: sudo apt install python3.11"
        exit 1
    fi
    ok "Python: $($PYTHON_BIN --version)"

    # venv
    if [ ! -d "venv" ]; then
        log "Создаю виртуальное окружение..."
        $PYTHON_BIN -m venv venv
        ok "venv создан"
    else
        ok "venv уже есть"
    fi

    PYTHON="$SCRIPT_DIR/venv/bin/python"
    PIP="$SCRIPT_DIR/venv/bin/pip"

    # Установка зависимостей
    if ! $PYTHON -c "import torch" 2>/dev/null; then
        log "Устанавливаю зависимости (CPU-only PyTorch)..."
        $PIP install --upgrade pip -q
        # CPU-only torch (~200 MB, значительно меньше GPU-версии)
        $PIP install torch --index-url https://download.pytorch.org/whl/cpu -q
        $PIP install numpy pandas matplotlib seaborn tqdm pyyaml imageio gymnasium -q
        # Устанавливаем сам пакет
        $PIP install -e . --no-deps -q 2>/dev/null || true
        ok "Зависимости установлены"
    else
        ok "torch $(${PYTHON} -c 'import torch; print(torch.__version__)') уже установлен"
    fi

    # Добавить src в путь чтобы находился пакет
    export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
}

# ─── 2. Параллельное обучение ─────────────────────────────────────────────────
run_training() {
    sep2; log "Шаг 2 / 4: Параллельное обучение"

    N_CPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    N_SEEDS=${#SEEDS[@]}
    THREADS=$(( N_CPU / N_SEEDS ))
    [ "$THREADS" -lt 1 ] && THREADS=1

    ok "CPU ядер: ${N_CPU} → ${N_SEEDS} процесса × ${THREADS} потоков = $((N_SEEDS * THREADS)) ядер"
    ok "Эпизодов на seed: ${EPISODES}  (всего: $((EPISODES * N_SEEDS)) эпизодов)"

    # Ограничиваем потоки чтобы процессы не мешали друг другу
    export OMP_NUM_THREADS=$THREADS
    export MKL_NUM_THREADS=$THREADS
    export OPENBLAS_NUM_THREADS=$THREADS
    export NUMEXPR_NUM_THREADS=$THREADS

    # warm-start: imitation checkpoint если есть
    INIT_CKPT=""
    if [ -f "outputs/models/imitation.pt" ]; then
        INIT_CKPT="outputs/models/imitation.pt"
        ok "Warm-start: ${INIT_CKPT}"
    else
        warn "Imitation checkpoint не найден — обучение с нуля"
    fi

    mkdir -p outputs/training outputs/models

    declare -A PIDS LOG_FILES

    sep
    log "Запускаю обучение..."

    for seed in "${SEEDS[@]}"; do
        OUT="outputs/training/seed_${seed}"
        mkdir -p "${OUT}/logs"
        LOG="$OUT/logs/stdout.log"
        LOG_FILES[$seed]="$LOG"

        CMD="$PYTHON scripts/train_dqn.py"
        CMD+=" --config configs/dqn.yaml"
        CMD+=" --episodes $EPISODES"
        CMD+=" --device cpu"
        CMD+=" --seed $seed"
        CMD+=" --output-dir $OUT"
        [ -n "$INIT_CKPT" ] && CMD+=" --init-checkpoint $INIT_CKPT"

        eval "$CMD" > "$LOG" 2>&1 &
        PIDS[$seed]=$!
        ok "seed=${seed}  PID=${PIDS[$seed]}  лог: $LOG"
    done

    sep
    log "Все процессы запущены. Ждём завершения..."
    log "(следить за прогрессом: tail -f outputs/training/seed_1/logs/stdout.log)"

    # Мониторинг каждые 2 минуты
    local elapsed=0
    while true; do
        sleep 120
        elapsed=$(( elapsed + 120 ))

        # Проверяем живые процессы
        local alive=0
        for seed in "${SEEDS[@]}"; do
            kill -0 "${PIDS[$seed]}" 2>/dev/null && alive=$(( alive + 1 ))
        done
        [ $alive -eq 0 ] && break

        # Прогресс из CSV логов
        echo
        log "── Прогресс через ${elapsed}s ─────────────────────────────────"
        for seed in "${SEEDS[@]}"; do
            local csv="outputs/training/seed_${seed}/logs/train_log.csv"
            if kill -0 "${PIDS[$seed]}" 2>/dev/null; then
                if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
                    local line; line=$(tail -1 "$csv")
                    local ep eps loss
                    ep=$(echo  "$line" | cut -d',' -f1)
                    eps=$(echo "$line" | cut -d',' -f3)
                    loss=$(echo "$line"| cut -d',' -f8)
                    echo "  seed=${seed}  ep=${ep}/${EPISODES}  ε=${eps}  loss=${loss}"
                else
                    echo "  seed=${seed}  [заполняется replay buffer...]"
                fi
            else
                echo "  seed=${seed}  [завершён]"
            fi
        done
    done

    # Финальная проверка
    sep
    local ok_count=0
    for seed in "${SEEDS[@]}"; do
        if wait "${PIDS[$seed]}" 2>/dev/null; then
            ok "seed=${seed} ✓"
            ok_count=$(( ok_count + 1 ))
        else
            fail "seed=${seed} упал (см. ${LOG_FILES[$seed]})"
        fi
    done
    log "${ok_count}/${N_SEEDS} обучений завершились успешно"
}

# ─── 3. Сохранить лучший checkpoint ──────────────────────────────────────────
save_best_checkpoint() {
    sep2; log "Шаг 3 / 4: Сохраняю лучший checkpoint"

    local best=""
    for seed in "${SEEDS[@]}"; do
        local ckpt="outputs/training/seed_${seed}/models/dqn_latest.pt"
        if [ -f "$ckpt" ]; then
            best="$ckpt"
            break
        fi
    done

    if [ -z "$best" ]; then
        warn "Ни один checkpoint не найден"
        return
    fi

    cp "$best" "outputs/models/dqn_best.pt"
    ok "outputs/models/dqn_best.pt  ← скопирован из $best"
}

# ─── 4. Финальный турнир ─────────────────────────────────────────────────────
run_evaluation() {
    sep2; log "Шаг 4 / 4: Финальный турнир"

    local ckpt="outputs/models/dqn_best.pt"
    [ ! -f "$ckpt" ] && { warn "Checkpoint не найден, турнир без DQN"; ckpt=""; }

    $PYTHON scripts/evaluate_agents.py \
        ${ckpt:+--checkpoint "$ckpt"} \
        --games "$EVAL_GAMES" \
        --device cpu \
        --seed 42 \
        --max-moves "$MAX_MOVES" \
        --out outputs/eval_final
}

# ─── Итоговый отчёт ──────────────────────────────────────────────────────────
print_report() {
    local T_END; T_END=$(date +%s)
    local ELAPSED=$(( T_END - T_START ))
    local MINS=$(( ELAPSED / 60 ))
    local SECS=$(( ELAPSED % 60 ))

    sep2
    echo -e "${C_GREEN}  ОБУЧЕНИЕ ЗАВЕРШЕНО${C_RESET}  (${MINS}м ${SECS}с)"
    sep2

    # Финальные метрики по seeds
    echo
    echo "  Финальные метрики обучения:"
    printf "  %-6s  %-8s  %-10s  %-12s  %-12s\n" \
        "seed" "episode" "epsilon" "loss" "|TD error|"
    echo "  ──────────────────────────────────────────────────"
    for seed in "${SEEDS[@]}"; do
        local csv="outputs/training/seed_${seed}/logs/train_log.csv"
        if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
            local row; row=$(tail -1 "$csv")
            local ep eps loss td
            ep=$(echo   "$row" | cut -d',' -f1)
            eps=$(echo  "$row" | cut -d',' -f3)
            loss=$(echo "$row" | cut -d',' -f8)
            td=$(echo   "$row" | cut -d',' -f17 2>/dev/null || echo "n/a")
            printf "  %-6s  %-8s  %-10s  %-12s  %-12s\n" \
                "$seed" "$ep" "$eps" "$loss" "$td"
        else
            printf "  %-6s  %s\n" "$seed" "[нет данных]"
        fi
    done

    # Результаты турнира
    local summary="outputs/eval_final/summary.csv"
    if [ -f "$summary" ]; then
        echo
        echo "  Финальный турнир (${EVAL_GAMES} игр на пару):"
        echo
        $PYTHON - "$summary" <<'PYEOF'
import csv, sys
path = sys.argv[1]
with open(path) as f:
    rows = list(csv.DictReader(f))
print(f"  {'агент':<14} {'игр':>6} {'побед':>7} {'win%':>8} {'draw%':>8} {'ходов':>8}")
print("  " + "─"*54)
for r in rows:
    name = r['agent']
    marker = " ◀" if name == "dqn" else ""
    print(f"  {name:<14} {int(r['games']):>6} {int(r['wins']):>7} "
          f"{float(r['win_rate']):>7.1%} "
          f"{float(r['draw_rate']):>7.1%} "
          f"{float(r['avg_moves']):>8.1f}{marker}")
PYEOF
    fi

    sep2
    echo "  Checkpoint:  outputs/models/dqn_best.pt"
    echo "  Логи:        outputs/training/seed_*/logs/train_log.csv"
    echo "  Турнир:      outputs/eval_final/summary.csv"
    sep2
    echo
}

# ─────────────────────────────────────────────────────────────────────────────
main() {
    echo
    sep2
    echo "  Уголки — DQN обучение"
    echo "  $(date)"
    sep2

    setup_env
    run_training
    save_best_checkpoint
    run_evaluation
    print_report
}

main "$@"
