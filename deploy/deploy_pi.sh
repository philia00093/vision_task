#!/bin/bash
# 一条命令把边缘端推上树莓派：代码 + 阶段样本 + 启动脚本 + 字体 + 重启 + 自检。
#
#   bash deploy/deploy_pi.sh              # 用默认主机（visionpi.local）
#   PI_HOST=10.105.58.57 bash deploy/deploy_pi.sh
#
# 幂等：重复跑只会同步有变化的东西，已经装过的字体/已经改过的启动脚本不会重做。
set -u

PI_HOST="${PI_HOST:-visionpi.local}"
PI_USER="${PI_USER:-wzk}"
PI_DIR="${PI_DIR:-/home/$PI_USER/vision_task}"
START_SH="/usr/local/sbin/vision-edge-start.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

say() { printf '\033[36m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[deploy] %s\033[0m\n' "$*" >&2; exit 1; }

ssh_pi() { ssh -o ConnectTimeout=6 -o BatchMode=yes "$PI_USER@$PI_HOST" "$@"; }

# ---------- 1. 能不能连上 ----------
# mDNS 名字时好时坏，先试名字，再退到固定 IP（PI_FALLBACK 可覆盖）
say "找树莓派（$PI_HOST / ${PI_FALLBACK:-10.105.58.57}）"
FOUND=""
for cand in "$PI_HOST" "${PI_FALLBACK:-10.105.58.57}"; do
    [ -z "$cand" ] && continue
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$PI_USER@$cand" true 2>/dev/null; then
        FOUND="$cand"; break
    fi
done
[ -z "$FOUND" ] && die "连不上（先确认同一网络 / ssh 可用）"
[ "$FOUND" != "$PI_HOST" ] && say "mDNS 没解析出来，改用 $FOUND"
PI_HOST="$FOUND"
say "目标 $PI_USER@$PI_HOST"

# ---------- 2. 代码 ----------
say "同步代码"
rsync -a --delete --exclude __pycache__ "$ROOT/edge/" "$PI_USER@$PI_HOST:$PI_DIR/edge/" || die "同步 edge/ 失败"
rsync -a "$ROOT/protocol.py" "$PI_USER@$PI_HOST:$PI_DIR/" || die "同步 protocol.py 失败"

# ---------- 3. 阶段参考图 ----------
say "同步阶段样本"
rsync -a --delete "$ROOT/vision_yolo_train/images/steps/" \
      "$PI_USER@$PI_HOST:$PI_DIR/vision_yolo_train/images/steps/" || die "同步样本失败"

# ---------- 4. 中文字体（缺了屏上一个字都不显示） ----------
if ssh_pi '[ -f /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc ]'; then
    say "中文字体已就绪"
else
    say "装中文字体（fonts-wqy-zenhei）"
    ssh_pi 'sudo apt-get install -y fonts-wqy-zenhei' >/dev/null 2>&1 \
        || say "  字体装失败，屏上标签会退回英文兜底（不影响识别）"
fi

# ---------- 5. 启动脚本参数（旧版是 --stage-interval，新版按秒） ----------
if ssh_pi "grep -q 'stage-period' $START_SH"; then
    say "启动脚本已是新参数"
else
    say "更新启动脚本参数"
    ssh_pi "sudo cp $START_SH $START_SH.bak-\$(date +%m%d-%H%M) && \
            sudo sed -i 's|--stage-interval 60 |--stage-period 0.5 --stage-window 1.0 |' $START_SH" \
        || die "改启动脚本失败"
fi

# ---------- 6. 重启 + 自检 ----------
say "重启 vision-edge"
ssh_pi 'sudo systemctl restart vision-edge' || die "重启失败"
sleep 18

say "最近状态"
ssh_pi 'journalctl -u vision-edge --since "-30s" --no-pager' \
    | grep -E "阶段识别\(SIFT\)|阶段识别\(近邻\)|\[阶段\]|\[fps\]|Traceback|Error" | tail -6
