#!/bin/bash

# Root文件合并脚本 - 处理合并过程中可能卡住的问题
# 使用方法: ./merge_root_files.sh

# 配置参数
source /lustre/collider/zhuxuliang/darkshine-simulation/setup.sh

WORK_DIR="/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn"
TRAIN_ROOT_DIR="${WORK_DIR}/Train_root_zejia"
MERGED_FILE="${TRAIN_ROOT_DIR}/Merged_Tracker_GNN.root"
TEMP_DIR="${WORK_DIR}/temp_merge"
LOG_FILE="${WORK_DIR}/merge_log_$(date +%Y%m%d_%H%M%S).txt"
BATCH_SIZE=2  # 每批次合并的文件数量
TIMEOUT=1800  # 每个合并步骤超时时间（秒）

# 日志函数
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# 清理函数
cleanup() {
    log "清理临时文件..."
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
        log "临时目录已删除"
    else
        log "临时目录不存在"
    fi
}

# 检查命令执行状态
check_status() {
    if [[ $? -ne 0 ]]; then
        log "错误: $1 失败"
        cleanup
        exit 1
    else
        log "$1 成功"
    fi
}

# 主函数
main() {
    log "开始合并Root文件..."
    
    # 创建临时目录
    mkdir -p "$TEMP_DIR"
    check_status "创建临时目录"
    
    # 获取所有源文件
    SOURCE_FILES=($(find "$TRAIN_ROOT_DIR" -name "Tracker_GNN_*.root" -type f | sort))
    FILE_COUNT=${#SOURCE_FILES[@]}
    
    if [[ $FILE_COUNT -eq 0 ]]; then
        log "错误: 没有找到源文件"
        cleanup
        exit 1
    fi
    
    log "找到 $FILE_COUNT 个源文件"
    
    # 初始化合并批次
    BATCH_NUM=0
    OUTPUT_FILES=()
    
    # 分批次合并文件
    log "开始分批次合并，每批次 $BATCH_SIZE 个文件"
    
    for ((i=0; i<$FILE_COUNT; i+=$BATCH_SIZE)); do
        BATCH_NUM=$((BATCH_NUM + 1))
        BATCH_FILES=()
        BATCH_FILES_STR=""
        
        # 获取当前批次的文件
        for ((j=i; j<i+BATCH_SIZE && j<FILE_COUNT; j++)); do
            BATCH_FILES+=("${SOURCE_FILES[$j]}")
            BATCH_FILES_STR+="${SOURCE_FILES[$j]} "
        done
        
        # 构建输出文件名
        BATCH_OUTPUT="${TEMP_DIR}/batch_${BATCH_NUM}.root"
        OUTPUT_FILES+=("$BATCH_OUTPUT")
        
        log "开始合并批次 $BATCH_NUM (${#BATCH_FILES[@]}个文件) 到 $BATCH_OUTPUT"
        log "批次文件: $BATCH_FILES_STR"
        
        # 执行合并命令（带超时）
        timeout $TIMEOUT hadd -f "$BATCH_OUTPUT" "${BATCH_FILES[@]}"
        CHECK_STATUS=$?
        
        if [[ $CHECK_STATUS -eq 124 ]]; then
            log "警告: 批次 $BATCH_NUM 合并超时 ($TIMEOUT 秒)"
            log "尝试恢复并继续..."
            
            # 尝试清理临时文件并重新合并
            rm -f "${BATCH_OUTPUT}.d*"
            timeout $TIMEOUT hadd -f "$BATCH_OUTPUT" "${BATCH_FILES[@]}"
            CHECK_STATUS=$?
            
            if [[ $CHECK_STATUS -eq 124 ]]; then
                log "错误: 批次 $BATCH_NUM 再次超时，跳过此批次"
                continue
            fi
        fi
        
        check_status "批次 $BATCH_NUM 合并"
        
        # 验证合并后的文件大小
        FILE_SIZE=$(du -sh "$BATCH_OUTPUT" | awk '{print $1}')
        log "批次 $BATCH_NUM 文件大小: $FILE_SIZE"
    done
    
    # 合并所有批次输出
    if [[ ${#OUTPUT_FILES[@]} -eq 0 ]]; then
        log "错误: 没有成功合并的批次"
        cleanup
        exit 1
    elif [[ ${#OUTPUT_FILES[@]} -eq 1 ]]; then
        log "只有一个批次，直接复制到最终输出"
        cp "${OUTPUT_FILES[0]}" "$MERGED_FILE"
    else
        log "开始合并所有批次文件到 $MERGED_FILE"
        hadd -f "$MERGED_FILE" "${OUTPUT_FILES[@]}"
        check_status "最终合并"
    fi
    
    # 验证最终合并文件
    # log "验证最终合并文件..."
    # root -l -b -q "$MERGED_FILE" 'gFile->Close();' &>> "$LOG_FILE"
    
    # if [[ $? -eq 0 ]]; then
    #     log "验证成功: $MERGED_FILE 可以正常打开"
    # else
    #     log "警告: $MERGED_FILE 可能存在问题，请检查"
    # fi
    
    # 清理临时文件
    cleanup
    
    log "合并过程完成！"
    log "最终文件: $MERGED_FILE"
    log "日志文件: $LOG_FILE"
    
    exit 0
}

# 执行主函数
main