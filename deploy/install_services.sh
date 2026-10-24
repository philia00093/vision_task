#!/usr/bin/env bash
# ============================================================================
# 安装两个开机自启服务：
#   vision-server   识别服务（8080，跑 best.pt，用 GPU）
#   vision-monitor  延时监控台（1000，只读上游，不占 GPU）
#
#   sudo bash deploy/install_services.sh
#
# 1000 是特权端口，所以 monitor 必须由 root 启动（这正是不用你手动跑的原因）。
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

die() { printf '\033[1;31m[错误] %s\033[0m\n' "$*" >&2; exit 1; }
say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || die "要以 root 运行：sudo bash deploy/install_services.sh"
[ -f "$PROJECT/server/run_server.py" ] || die "找不到 $PROJECT/server/run_server.py"
[ -f "$PROJECT/vision_yolo_train/runs/wersion_parts/weights/best.pt" ] || \
  die "找不到 best.pt：$PROJECT/vision_yolo_train/runs/wersion_parts/weights/best.pt"

say "停掉手工起的进程（避免抢 8080/1000）"
fuser -k 8080/tcp 2>/dev/null || true
fuser -k 2000/tcp 2>/dev/null || true
pkill -f "server/[r]un_server[.]py" 2>/dev/null || true
sleep 1

say "安装 unit"
install -m 0644 "$HERE/vision-server.service"  /etc/systemd/system/vision-server.service
install -m 0644 "$HERE/vision-monitor.service" /etc/systemd/system/vision-monitor.service
systemctl daemon-reload

say "启动并设为开机自启"
systemctl enable --now vision-server.service
sleep 2
systemctl enable --now vision-monitor.service
sleep 3

say "状态"
systemctl --no-pager --lines=3 status vision-server.service | head -8
echo
systemctl --no-pager --lines=3 status vision-monitor.service | head -8

IP=$(hostname -I | awk '{print $1}')
say "完成"
echo "  识别服务      http://$IP:8080/"
echo "  延时监控台    http://$IP:2000/     ← 看推理耗时/帧龄/帧率，以及树莓派传过来的画面"
echo "  手机上也能打开（同一热点）"
