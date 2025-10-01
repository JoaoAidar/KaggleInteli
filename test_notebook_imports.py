"""
Test that all notebook imports work correctly.
"""
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("TESTING NOTEBOOK IMPORTS")
print("="*70)

# Simulate notebook environment
notebook_dir = os.path.join(os.getcwd(), 'notebooks')
parent_dir = os.path.dirname(notebook_dir)
sys.path.insert(0, parent_dir)

print(f"\nPython path configured:")
print(f"  - Notebook dir: {notebook_dir}")
print(f"  - Parent dir: {parent_dir}")
print(f"  - sys.path[0]: {sys.path[0]}")

# Test standard library imports
print("\n1. Testing standard library imports...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import StratifiedKFold
    print("   ✓ All standard libraries imported successfully")
    print(f"     - numpy: {np.__version__}")
    print(f"     - pandas: {pd.__version__}")
    print(f"     - matplotlib: {plt.matplotlib.__version__}")
    print(f"     - seaborn: {sns.__version__}")
except Exception as e:
    print(f"   ✗ Error importing standard libraries: {e}")
    sys.exit(1)

# Test custom module imports
print("\n2. Testing custom module imports...")
try:
    from src.io_utils import load_data, get_target_name, save_submission
    from src.features import split_columns, build_preprocessor
    from src.modeling import build_pipelines, random_search_rf
    from src.evaluation import evaluate_all, cv_report, assert_min_accuracy
    print("   ✓ All custom modules imported successfully")
    print("     - src.io_utils: 3 functions")
    print("     - src.features: 2 functions")
    print("     - src.modeling: 2 functions")
    print("     - src.evaluation: 3 functions")
except Exception as e:
    print(f"   ✗ Error importing custom modules: {e}")
    sys.exit(1)

# Test that functions are callable
print("\n3. Testing function signatures...")
try:
    # Test load_data
    import inspect
    sig = inspect.signature(load_data)
    print(f"   ✓ load_data{sig}")
    
    sig = inspect.signature(split_columns)
    print(f"   ✓ split_columns{sig}")
    
    sig = inspect.signature(build_pipelines)
    print(f"   ✓ build_pipelines{sig}")
    
    sig = inspect.signature(evaluate_all)
    print(f"   ✓ evaluate_all{sig}")
except Exception as e:
    print(f"   ✗ Error checking function signatures: {e}")
    sys.exit(1)

# Test matplotlib configuration
print("\n4. Testing matplotlib configuration...")
try:
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    print("   ✓ Matplotlib style configured successfully")
except Exception as e:
    print(f"   ⚠ Warning: Could not set matplotlib style: {e}")
    print("   (This is not critical - default style will be used)")

print("\n" + "="*70)
print("ALL IMPORT TESTS PASSED!")
print("="*70)
print("\nThe notebook should be able to import all required libraries.")
print("If you still experience issues in Jupyter, try:")
print("  1. Restart the Jupyter kernel")
print("  2. Run: %matplotlib inline")
print("  3. Ensure you're running from the notebooks/ directory")

