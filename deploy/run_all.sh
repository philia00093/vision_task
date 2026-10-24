#!/usr/bin/env bash
# ============================================================================
# 一键起整套系统（在笔记本上跑；树莓派那边由它 ssh 过去重启）
#
#   bash deploy/run_all.sh              # 起 识别服务 + 监控台 + 树莓派末端
#   bash deploy/run_all.sh --status     # 只看状态，不动进程
#   bash deploy/run_all.sh --stop       # 全停
#
# 环境变量：
#   PI_HOST=visionpi.local   树莓派地址（默认 mDNS 名，也可以写 IP）
#   WEIGHTS=...              权重文件，默认 best.pt（v2 过拟合已弃用）
# ============================================================================
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HOME/miniforge3/envs/yolo26/bin/python"
PI_HOST="${PI_HOST:-visionpi.local}"
WEIGHTS="${WEIGHTS:-$ROOT/vision_yolo_train/runs/wersion_parts/weights/best.pt}"

bold() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
busy() { ss -ltnH "sport = :$1" 2>/dev/null | grep -q .; }

ip_addr() { hostname -I | awk '{print $1}'; }

# 树莓派 SSH：先按名字试两次，再解析成 IP 带 accept-new 试（Wi-Fi 抖动 + host key 两个坑）
pi_ssh() {
    local cmd="$1" attempt host
    for attempt in 1 2 3; do
        if timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 "wzk@$PI_HOST" "$cmd" 2>/dev/null; then
            return 0
        fi
        host=$(getent hosts "$PI_HOST" 2>/dev/null | awk 'NR==1{print $1}')
        if [ -n "$host" ] && timeout 12 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
                -o ConnectTimeout=8 "wzk@$host" "$cmd" 2>/dev/null; then
            return 0
        fi
        sleep 2
    done
    return 1
}



# ---------------------------------------------------------------- 启动
start_server() {
    if busy 8080; then ok "识别服务已在跑（8080）"; return; fi
    [ -f "$WEIGHTS" ] || { bad "找不到权重 $WEIGHTS"; return 1; }
    setsid nohup "$PY" -u "$ROOT/server/run_server.py" \
        --host 0.0.0.0 --port 8080 --imgsz 640 --device 0 --window 8 \
        --weights "$WEIGHTS" > /tmp/server.log 2>&1 < /dev/null &
    disown
    for _ in $(seq 1 30); do busy 8080 && break; sleep 1; done
    busy 8080 && ok "识别服务已启动（8080，$(basename "$WEIGHTS")）" \
              || { bad "识别服务没起来，看 /tmp/server.log"; tail -5 /tmp/server.log; }
}

start_monitor() {
    if busy 2000; then ok "监控台已在跑（2000）"; return; fi
    setsid nohup python3 -u "$ROOT/server/monitor.py" \
        --host 0.0.0.0 --port 2000 --upstream http://127.0.0.1:8080 \
        > /tmp/monitor.log 2>&1 < /dev/null &
    disown
    for _ in $(seq 1 10); do busy 2000 && break; sleep 1; done
    busy 2000 && ok "监控台已启动（2000）" || bad "监控台没起来，看 /tmp/monitor.log"
}

start_pi() {
    if ! pi_ssh true; then
        bad "连不上树莓派 $PI_HOST（开机了吗？和笔记本在同一个热点吗？）"
        return
    fi
    pi_ssh 'sudo systemctl restart vision-edge'
    sleep 12
    state=$(pi_ssh 'systemctl is-active vision-edge')
    [ "$state" = "active" ] && ok "树莓派末端已启动（$PI_HOST）" || bad "树莓派末端状态：$state"
}

# ---------------------------------------------------------------- 状态
show_status() {
    if busy 8080; then
        ok "识别服务 8080 在跑"
        curl -sS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | head -c 200; echo
        curl -sS --max-time 3 http://127.0.0.1:8080/state 2>/dev/null | "$PY" -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('  （还没有帧）'); raise SystemExit
print(f'  帧号 {d.get(\"count\")} | 推理 {d.get(\"ms\")} ms | 画面 {d.get(\"size\")} | 框 {len(d.get(\"boxes\", []))}')
" 2>/dev/null
    else
        bad "识别服务 8080 没在跑"
    fi
    busy 2000 && ok "监控台 2000 在跑" || bad "监控台 2000 没在跑"
    if pi_ssh true; then
        st=$(pi_ssh 'systemctl is-active vision-edge')
        [ "$st" = "active" ] && ok "树莓派末端 active" || bad "树莓派末端 $st"
        pi_ssh 'sudo journalctl -u vision-edge --since "-30 s" --no-pager 2>/dev/null | grep "\[fps\]" | tail -1' | sed 's/^/  /'
    else
        bad "连不上树莓派 $PI_HOST"
    fi
    echo
    echo "  监控台 http://$(ip_addr):2000/"
    echo "  识别服务 http://$(ip_addr):8080/"
}

# ---------------------------------------------------------------- 停止
stop_all() {
    fuser -k 8080/tcp 2>/dev/null && ok "识别服务已停" || ok "识别服务本来就没跑"
    fuser -k 2000/tcp 2>/dev/null && ok "监控台已停" || ok "监控台本来就没跑"
    if pi_ssh 'sudo systemctl stop vision-edge'; then
        ok "树莓派末端已停"
    else
        bad "连不上树莓派，末端可能还在跑"
    fi
}

case "${1:-}" in
    --status) show_status ;;
    --stop)   bold "停止"; stop_all ;;
    *)
        bold "启动"
        start_server
        start_monitor
        start_pi
        show_status
        ;;
esac
