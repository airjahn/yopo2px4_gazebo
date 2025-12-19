#!/bin/bash

#================================================================#
#           无人机仿真与YOPO系统自动化启动脚本 (V2)
#
# 功能:
# 1. 启动仿真器 (base)
# 2. 启动 takeoff.py (yopo)
# 3. 监控 takeoff.py 直到它输出 "OFFBOARD enabled" 和 "Vehicle armed"
# 4. **杀死** takeoff.py
# 5. 启动 yopo_mvpa.py, test4.py (yopo) 和 RViz (base)
#================================================================#

# --- 配置 ---
# 假设此脚本在 ~/YOPO/YOPO 目录中运行
SIM_LAUNCH_FILE="$HOME/DroneSys/Simulator/gazebo_simulator/launch_basic/sitl.launch"
RVIZ_CONFIG_FILE="yopomvp.rviz" # 相对路径
TAKEOFF_LOG="takeoff.log"     # 用于监控 takeoff.py 的临时日志

# --- 清理函数 ---
pids=() # 存储所有后台进程ID
TAKEOFF_PID="" # 单独存储 takeoff.py 的PID

cleanup() {
    echo -e "\n\n[INFO] 收到退出信号... 正在关闭所有后台进程..."
    
    # 倒序关闭所有在 pids 数组中的进程
    # 使用 sort -r 来获取倒序的索引
    indices=($(seq 0 $((${#pids[@]} - 1)) | sort -r))
    for i in "${indices[@]}"; do
        pid=${pids[$i]}
        echo "[INFO] 正在停止 PID: $pid"
        kill -SIGINT "$pid" 2>/dev/null
    done
    
    # 额外确保ros核心进程被关闭
    pkill -f 'rosmaster'
    pkill -f 'gzserver'
    pkill -f 'gzclient'
    
    echo "[INFO] 清理临时文件 $TAKEOFF_LOG..."
    rm -f $TAKEOFF_LOG
    
    echo "[INFO] 清理完成。再见！"
    exit 0
}

# 捕获 Ctrl+C 信号
trap cleanup SIGINT

# --- 实用函数 ---
# 函数: 监控 takeoff.py 的日志
# $1: 要等待的字符串
# $2: takeoff.py 的PID
wait_for_log() {
    local log_string="$1"
    local pid_to_watch="$2"
    echo "[INFO] 正在等待日志 (PID: $pid_to_watch): '$log_string'..."
    
    while ! grep -q "$log_string" $TAKEOFF_LOG; do
        # 检查 takeoff.py 进程是否意外崩溃
        if ! kill -0 "$pid_to_watch" 2>/dev/null; then
            echo -e "\n[ERROR] takeoff.py (PID: $pid_to_watch) 进程意外终止！"
            echo "[ERROR] 请检查 $TAKEOFF_LOG 获取详情。"
            cleanup
            exit 1
        fi
        sleep 0.5
    done
    
    echo "[INFO] '$log_string' 确认！"
}

# --- 主脚本 ---

# 步骤 0: 初始化 Conda
echo "[INFO] 正在初始化 Conda..."
eval "$(conda shell.bash hook)"
if [ $? -ne 0 ]; then
    echo "[ERROR] Conda 初始化失败。"
    exit 1
fi

# 清理上次的日志
rm -f $TAKEOFF_LOG

# 步骤 1: (base) 启动 Gazebo 仿真
echo "[INFO] 启动 Gazebo SITL ($SIM_LAUNCH_FILE)..."
conda run -n base roslaunch $SIM_LAUNCH_FILE &
ROSLAUNCH_PID=$!
pids+=($ROSLAUNCH_PID)
echo "[INFO] Gazebo 进程已启动 (PID: $ROSLAUNCH_PID)."

# 步骤 2: 等待仿真加载 (检查ROS服务)
SERVICE_TO_CHECK="/mavros/cmd/arming"
echo "[INFO] 正在等待仿真完全加载 (检查服务: $SERVICE_TO_CHECK)..."
until conda run -n base rosservice list 2>/dev/null | grep -q "$SERVICE_TO_CHECK"; do
    echo -n "."
    sleep 1
    # 检查roslaunch是否意外退出
    if ! kill -0 $ROSLAUNCH_PID 2>/dev/null; then
        echo -e "\n[ERROR] roslaunch 进程 (PID: $ROSLAUNCH_PID) 意外终止！"
        cleanup
        exit 1
    fi
done
echo -e "\n[INFO] 仿真已就绪！"

# 步骤 3: (yopo) 启动 takeoff.py (在后台，并重定向输出)
echo "[INFO] 启动 takeoff.py (在后台) 并监控其日志..."
# 将 stdout 和 stderr 都重定向到日志文件
conda run -n yopo python takeoff.py > $TAKEOFF_LOG 2>&1 &
TAKEOFF_PID=$!
pids+=($TAKEOFF_PID) # 添加到pids数组，以便Ctrl+C时能清理
echo "[INFO] takeoff.py 正在运行 (PID: $TAKEOFF_PID)。日志: $TAKEOFF_LOG"

# 步骤 4: 监控 takeoff.py 的输出
wait_for_log "OFFBOARD enabled" $TAKEOFF_PID
wait_for_log "Vehicle armed" $TAKEOFF_PID

# 步骤 5: 杀死 takeoff.py
echo "[INFO] takeoff.py 已完成其任务。正在终止 (PID: $TAKEOFF_PID)..."
kill -SIGINT $TAKEOFF_PID
sleep 1 # 等待进程响应

# 确认杀死
if kill -0 $TAKEOFF_PID 2>/dev/null; then
    echo "[INFO] takeoff.py 未立即退出，强制杀死 (kill -9)..."
    kill -9 $TAKEOFF_PID
fi
echo "[INFO] takeoff.py 已终止。"

# 从pids数组中移除 TAKEOFF_PID，因为它已经被处理了
# (Bash 数组移除语法)
new_pids=()
for pid in "${pids[@]}"; do
    if [ "$pid" != "$TAKEOFF_PID" ]; then
        new_pids+=("$pid")
    fi
done
pids=("${new_pids[@]}")


# 步骤 6: (yopo & base) 启动所有分析和可视化程序
echo "[INFO] 达到目标状态。正在启动 yopo_mvpa.py, test4.py 和 RViz..."

echo "[CMD] conda run -n yopo python yopo_mvpa.py"
conda run -n yopo python yopo_mvpa.py &
pids+=($!) # 添加PID

echo "[CMD] conda run -n yopo python test4.py"
conda run -n yopo python test4.py &
pids+=($!) # 添加PID

echo "[CMD] conda run -n base rviz -d $RVIZ_CONFIG_FILE"
conda run -n base rviz -d $RVIZ_CONFIG_FILE &
pids+=($!) # 添加PID

echo -e "\n[SUCCESS] 所有后续进程已启动。"
echo "============================================="
echo "      系统正在运行中..."
echo "      按下 [Ctrl+C] 来关闭所有剩余进程。"
echo "============================================="

# 等待所有剩余的后台任务
wait