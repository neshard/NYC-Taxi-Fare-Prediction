#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline runner untuk NYC Taxi Fair Prediction

Script ini menjalankan ketiga tahap secara berurutan:
1. 04_modeling.py   - Feature engineering & training model
2. 05_evaluation.py - Evaluasi model
3. 06_tuning.py     - Hyperparameter tuning

Usage:
    python run_pipeline.py

    atau jalankan individual:
    python 04_modeling.py
    python 05_evaluation.py
    python 06_tuning.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Warna output
class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*70}{Color.ENDC}")
    print(f"{Color.BOLD}{Color.BLUE}{text:^70}{Color.ENDC}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*70}{Color.ENDC}\n")

def print_success(text):
    print(f"{Color.GREEN}✅ {text}{Color.ENDC}")

def print_warning(text):
    print(f"{Color.YELLOW}⚠️  {text}{Color.ENDC}")

def print_error(text):
    print(f"{Color.RED}❌ {text}{Color.ENDC}")

def run_script(script_name):
    """Run individual script"""
    script_path = Path(script_name)
    
    if not script_path.exists():
        print_error(f"Script tidak ditemukan: {script_name}")
        return False
    
    print_header(f"Running {script_name}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path.cwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print_success(f"{script_name} selesai dengan sukses")
            return True
        else:
            print_error(f"{script_name} gagal dengan return code {result.returncode}")
            return False
    
    except Exception as e:
        print_error(f"Error menjalankan {script_name}: {e}")
        return False

def main():
    print_header("NYC Taxi Fair Prediction Pipeline")
    
    scripts = [
        "04_modeling.py",
        "05_evaluation.py",
        "06_tuning.py"
    ]
    
    print(f"Scripts yang akan dijalankan:")
    for i, script in enumerate(scripts, 1):
        print(f"  {i}. {script}")
    
    print("\n" + "="*70)
    
    results = {}
    for script in scripts:
        success = run_script(script)
        results[script] = success
        
        if not success:
            print_warning(f"Pipeline dihentikan di {script}")
            break
    
    # Summary
    print_header("Pipeline Summary")
    
    for script, success in results.items():
        status = "✅ Selesai" if success else "❌ Gagal"
        print(f"{script:<30} {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print_success("Pipeline selesai dengan sukses!")
        print(f"\n{Color.CYAN}Output tersimpan di:{Color.ENDC}")
        print(f"  - Models: {Color.BOLD}models/{Color.ENDC}")
        print(f"  - Results: {Color.BOLD}outputs/{Color.ENDC}")
        print(f"  - Data: {Color.BOLD}data/{Color.ENDC}")
    else:
        print_error("Pipeline tidak selesai dengan baik. Lihat error di atas.")
        sys.exit(1)

if __name__ == "__main__":
    main()
