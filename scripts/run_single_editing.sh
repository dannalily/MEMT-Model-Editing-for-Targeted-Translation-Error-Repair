#!/bin/bash

# Conda environment setup

# Create directories
mkdir -p logs_editing logs_editing_eval logs_script

# Define methods, GPUs, and language arrays
methods=(FT ROME MEMIT AlphaEdit UNKE WISE GRACE)
gpus=(0 1 2 3 4 5 6)
slangs=(en zh)
tlangs=(ja fr de ar)

declare -A METHODS_en=(
    ["FT"]="layer7"
    ["ROME"]="layer7"
    ["MEMIT"]="layer5_6_7_8_9"
    ["AlphaEdit"]="layer5_6_7_8_9"
    ["UNKE"]="layer7"
    ["WISE"]="layer30"
    ["GRACE"]="layer30"
)

declare -A METHODS_zh=(
    ["FT"]="layer5"
    ["ROME"]="layer5"
    ["MEMIT"]="layer2_3_4_5_6"
    ["AlphaEdit"]="layer2_3_4_5_6"
    ["UNKE"]="layer5"
    ["WISE"]="layer30"
    ["GRACE"]="layer30"
)

# Master script log
SCRIPT_LOG="logs_script/script_single_editing.log"

# Redirect all echo output to log file AND terminal
exec > >(tee "$SCRIPT_LOG") 2>&1

echo "=========================================="
echo "Script started at: $(date)"
echo "Log file: $SCRIPT_LOG"
echo "=========================================="
echo ""

# Execute each method sequentially for all language combinations
for slang in "${slangs[@]}"; do
    for tlang in "${tlangs[@]}"; do
        
        echo ""
        echo "=========================================="
        echo "Language Pair: $slang -> $tlang"
        echo "Started at: $(date)"
        echo "=========================================="
        
        # EDITING PHASE
        echo ""
        echo "--- EDITING PHASE ---"
        
        # Array to store PIDs
        pids_editing=()
        
        for i in "${!methods[@]}"; do
            method=${methods[$i]}
            gpu=${gpus[$i]}
            mkdir -p logs_editing/${method,,}
            
            echo "Starting editing: $method on GPU $gpu"
            
            CUDA_VISIBLE_DEVICES=$gpu python -u editing_code/edit_main.py \
                --editing_method $method \
                --hparam_file editing_code/hparams/$method/qwen2-5-3b-instruct_${slang}2x.yaml \
                --slang $slang \
                --tlang $tlang \
                > logs_editing/${method,,}/qwen3b_${slang}_${tlang}.txt 2>&1 &
            
            pids_editing+=($!)
            echo "  → PID: $! (${method})"
        done
        
        echo "Waiting for all editing jobs to complete..."
        echo "PIDs: ${pids_editing[@]}"
        
        # Wait for each PID and check status
        for pid in "${pids_editing[@]}"; do
            if wait $pid; then
                echo "  ✓ PID $pid completed successfully"
            else
                echo "  ✗ PID $pid failed with exit code $?"
            fi
        done
        
        echo "✓ All Editing Jobs Completed: $slang -> $tlang"
        
        # EVALUATION PHASE
        echo ""
        echo "--- EVALUATION PHASE ---"
        
        # Array to store PIDs
        pids_eval=()
        
        for i in "${!methods[@]}"; do
            method=${methods[$i]}
            gpu=${gpus[$i]}
            if [ "$slang" == "en" ]; then
                layer="${METHODS_en[$method]}"
            elif [ "$slang" == "zh" ]; then
                layer="${METHODS_zh[$method]}"
            fi
            
            mkdir -p logs_editing_eval/${method,,}
            echo "Starting evaluation: $method on GPU $gpu (layer: $layer)"
            
            CUDA_VISIBLE_DEVICES=$gpu python -u evaluation_editing.py \
                --metric bleurt \
                --editing $method \
                --annotate $layer \
                --slang $slang \
                --tlang $tlang \
                --translator Qwen2-5-3B-Instruct \
                --type single \
                > logs_editing_eval/${method,,}/qwen3b_${slang}_${tlang}.txt 2>&1 &
            
            pids_eval+=($!)
            echo "  → PID: $! (${method})"
        done
        
        echo "Waiting for all evaluation jobs to complete..."
        echo "PIDs: ${pids_eval[@]}"
        
        # Wait for each PID and check status
        for pid in "${pids_eval[@]}"; do
            if wait $pid; then
                echo "  ✓ PID $pid completed successfully"
            else
                echo "  ✗ PID $pid failed with exit code $?"
            fi
        done
        
        echo "✓ All Evaluation Jobs Completed: $slang -> $tlang"

        # LANGUAGE CHECK
        echo ""
        echo "--- LANGUAGE CHECK ---"
        echo "Starting language check for $slang -> $tlang"

        python language_detection.py \
                --editing_type single \
                --editing_slang $slang \
                --editing_tlang $tlang \
                > logs_editing_eval/language_${slang}_${tlang}.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ Language Check Finished: $slang -> $tlang"
        else
            echo "✗ Language Check Failed: $slang -> $tlang"
        fi
        
        # EXCEL GENERATION
        echo ""
        echo "--- EXCEL GENERATION ---"
        echo "Generating Excel results for $slang -> $tlang"
        
        python print_single_editing_eval_results_excel.py \
            --slang $slang \
            --tlang $tlang \
            > logs_editing_eval/excel_${slang}_${tlang}.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ Excel Results Generated: $slang -> $tlang"
        else
            echo "✗ Excel Results Failed: $slang -> $tlang"
        fi
        
        echo ""
        echo "✓ Language Pair Completed: $slang -> $tlang at $(date)"
        echo "=========================================="
        
    done
done

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
