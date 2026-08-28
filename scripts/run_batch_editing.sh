#!/bin/bash

# Conda environment setup

# Create log directories
mkdir -p logs_batch_editing logs_batch_editing_eval logs_script

# Define methods, GPUs, seeds, and language arrays
methods=(FT MEMIT AlphaEdit UNKE)
gpus=(0 1 2 3 4 5 6 7)
seeds=(42 0 1 1234 10 123 2 5)
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
SCRIPT_LOG="logs_script/script_batch_editing.log"

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
        
        for method in "${methods[@]}"; do
            # Array to store PIDs
            pids_editing=()
            mkdir -p logs_batch_editing/${method,,}/qwen3b_${slang}_${tlang}

            for i in "${!seeds[@]}"; do
                seed=${seeds[$i]}
                gpu=${gpus[$i]}
            
                echo "Starting batch editing: $method with seed $seed on GPU $gpu "
                
                CUDA_VISIBLE_DEVICES=$gpu python -u editing_code/edit_batch_main.py \
                    --editing_method $method \
                    --hparam_file editing_code/hparams/$method/qwen2-5-3b-instruct_${slang}2x.yaml \
                    --slang $slang \
                    --tlang $tlang \
                    --seed $seed \
                    > logs_batch_editing/${method,,}/qwen3b_${slang}_${tlang}/${seed}.txt 2>&1 &
                pids_editing+=($!)
                echo "  → PID: $! (${method})"
            done
            echo "Waiting $method with all seeds to complete..."
            echo "PIDs: ${pids_editing[@]}"
            
            # Wait for each PID and check status
            for pid in "${pids_editing[@]}"; do
                if wait $pid; then
                    echo "  ✓ PID $pid completed successfully"
                else
                    echo "  ✗ PID $pid failed with exit code $?"
                fi
            done
        done
        
        echo "✓ All Editing Jobs Completed: $slang -> $tlang"
        
        # EVALUATION PHASE
        echo ""
        echo "--- EVALUATION PHASE ---"
            
        for method in "${methods[@]}"; do
            if [ "$slang" == "en" ]; then
                layer="${METHODS_en[$method]}"
            elif [ "$slang" == "zh" ]; then
                layer="${METHODS_zh[$method]}"
            fi

            mkdir -p logs_batch_editing_eval/${method,,}qwen3b_${slang}_${tlang}
            pids_eval=()

            for i in "${!seeds[@]}"; do
                seed=${seeds[$i]}
                gpu=${gpus[$i]}

                echo "Starting batch editing evaluation: $method with seed $seed on GPU $gpu"
            
                CUDA_VISIBLE_DEVICES=$gpu python -u evaluation_editing.py \
                    --metric bleurt \
                    --editing $method \
                    --annotate $layer \
                    --slang $slang \
                    --tlang $tlang \
                    --translator Qwen2-5-3B-Instruct \
                    --type batch \
                    --seed $seed \
                    > logs_batch_editing_eval/${method,,}qwen3b_${slang}_${tlang}/${seed}.txt 2>&1 &
            
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
        done
        
        echo "✓ All Evaluation Jobs Completed: $slang -> $tlang"

        # LANGUAGE CHECK
        echo ""
        echo "--- LANGUAGE CHECK ---"
        echo "Starting language check for $slang -> $tlang"

        python language_detection.py \
                --editing_type batch \
                --editing_slang $slang \
                --editing_tlang $tlang \
                > logs_batch_editing_eval/language_${slang}_${tlang}.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ Language Check Finished: $slang -> $tlang"
        else
            echo "✗ Language Check Failed: $slang -> $tlang"
        fi
        
        # EXCEL GENERATION
        echo ""
        echo "--- EXCEL GENERATION ---"
        echo "Generating Excel results for $slang -> $tlang"
        
        python print_seq_bat_editing_eval_results_excel.py \
            --slang $slang \
            --tlang $tlang \
            --editing-type batch \
            --output-dir excel_batch_editing \
            > logs_batch_editing_eval/excel_editing_${slang}_${tlang}.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ Excel Editing Results Generated: $slang -> $tlang"
        else
            echo "✗ Excel Editing Failed: $slang -> $tlang"
        fi

        # PLOT GENERATION
        echo ""
        echo "--- PLOT GENERATION ---"
        echo "Generating PLOT for $slang -> $tlang"
        
        python print_seq_bat_editing_eval_results_plot.py \
            --slang $slang \
            --tlang $tlang \
            --editing-type batch \
             --input_dir excel_batch_editing \
             --output_dir plot_batch_editing \
            > logs_batch_editing_eval/plot_editing_${slang}_${tlang}.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ PLOT Editing Generated: $slang -> $tlang"
        else
            echo "✗ PLOT Editing Failed: $slang -> $tlang"
        fi

        # EXCEL LANGUAGE GENERATION
        echo ""
        echo "--- EXCEL LANGUAGE GENERATION ---"
        echo "Generating Excel results for $slang -> $tlang"
        
        python print_seq_bat_editing_lanugage_excel.py \
            --slang $slang \
            --tlang $tlang \
            --editing-type batch \
            --output-dir excel_batch_language \
            > logs_batch_editing_eval/excel_language_${slang}_${tlang}.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ Excel Language Generated: $slang -> $tlang"
        else
            echo "✗ Excel Language Failed: $slang -> $tlang"
        fi

        # PLOT GENERATION
        echo ""
        echo "--- PLOT LANGUAGE GENERATION ---"
        echo "Generating PLOT for $slang -> $tlang"
        
        python print_seq_bat_editing_eval_results_plot.py \
            --slang $slang \
            --tlang $tlang \
            --metric language \
            --editing-type batch \
             --input_dir excel_batch_language \
             --output_dir plot_batch_language \
            > logs_batch_editing_eval/plot_language_${slang}_${tlang}.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ PLOT Language Generated: $slang -> $tlang"
        else
            echo "✗ PLOT Language Failed: $slang -> $tlang"
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
