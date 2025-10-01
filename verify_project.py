"""
Project verification script.
Checks that all required files exist and are properly structured.
"""
import os
import json
import sys

def check_file(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (MISSING)")
        return False

def check_directory(path, description):
    """Check if a directory exists."""
    if os.path.isdir(path):
        print(f"✓ {description}: {path}/")
        return True
    else:
        print(f"✗ {description}: {path}/ (MISSING)")
        return False

def main():
    """Run verification checks."""
    print("="*60)
    print("PROJECT VERIFICATION")
    print("="*60)
    
    all_checks = []
    
    # Check directories
    print("\n📁 Checking Directories...")
    all_checks.append(check_directory("data", "Data directory"))
    all_checks.append(check_directory("src", "Source code directory"))
    all_checks.append(check_directory("notebooks", "Notebooks directory"))
    all_checks.append(check_directory("reports", "Reports directory"))
    
    # Check data files (user-provided)
    print("\n📊 Checking Data Files...")
    all_checks.append(check_file("data/train.csv", "Training data"))
    all_checks.append(check_file("data/test.csv", "Test data"))
    all_checks.append(check_file("data/sample_submission.csv", "Sample submission"))
    
    # Check source files
    print("\n🐍 Checking Source Files...")
    all_checks.append(check_file("src/__init__.py", "Package init"))
    all_checks.append(check_file("src/io_utils.py", "I/O utilities"))
    all_checks.append(check_file("src/features.py", "Feature engineering"))
    all_checks.append(check_file("src/modeling.py", "Model building"))
    all_checks.append(check_file("src/evaluation.py", "Evaluation utilities"))
    all_checks.append(check_file("src/cli.py", "CLI interface"))
    
    # Check notebook
    print("\n📓 Checking Notebook...")
    notebook_exists = check_file("notebooks/01_startup_success.ipynb", "Main notebook")
    all_checks.append(notebook_exists)
    
    if notebook_exists:
        try:
            with open("notebooks/01_startup_success.ipynb", 'r') as f:
                notebook = json.load(f)
                num_cells = len(notebook.get('cells', []))
                print(f"  → Notebook has {num_cells} cells")
                if num_cells >= 50:
                    print(f"  ✓ Sufficient cells (expected ~55)")
                else:
                    print(f"  ⚠ Warning: Expected ~55 cells, found {num_cells}")
        except Exception as e:
            print(f"  ✗ Error reading notebook: {e}")
            all_checks.append(False)
    
    # Check documentation
    print("\n📄 Checking Documentation...")
    all_checks.append(check_file("README.md", "README"))
    all_checks.append(check_file("Makefile", "Makefile"))
    all_checks.append(check_file("PROJECT_SUMMARY.md", "Project summary"))
    
    # Check generated files (optional)
    print("\n📈 Checking Generated Files (optional)...")
    if os.path.exists("reports/cv_metrics.csv"):
        print("✓ CV metrics file exists: reports/cv_metrics.csv")
    else:
        print("ℹ CV metrics not yet generated (run 'make cv')")
    
    if os.path.exists("submission.csv"):
        print("✓ Submission file exists: submission.csv")
        # Check format
        with open("submission.csv", 'r') as f:
            first_line = f.readline().strip()
            if first_line == "id,labels":
                print("  ✓ Correct header format")
            else:
                print(f"  ✗ Incorrect header: {first_line}")
    else:
        print("ℹ Submission not yet generated (run 'make submit')")
    
    # Test imports
    print("\n🔧 Testing Module Imports...")
    try:
        from src.io_utils import load_data, get_target_name, save_submission
        print("✓ io_utils imports successfully")
        all_checks.append(True)
    except Exception as e:
        print(f"✗ io_utils import failed: {e}")
        all_checks.append(False)
    
    try:
        from src.features import split_columns, build_preprocessor
        print("✓ features imports successfully")
        all_checks.append(True)
    except Exception as e:
        print(f"✗ features import failed: {e}")
        all_checks.append(False)
    
    try:
        from src.modeling import build_pipelines, random_search_rf
        print("✓ modeling imports successfully")
        all_checks.append(True)
    except Exception as e:
        print(f"✗ modeling import failed: {e}")
        all_checks.append(False)
    
    try:
        from src.evaluation import evaluate_all, cv_report, assert_min_accuracy
        print("✓ evaluation imports successfully")
        all_checks.append(True)
    except Exception as e:
        print(f"✗ evaluation import failed: {e}")
        all_checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"\nChecks passed: {passed}/{total} ({percentage:.1f}%)")
    
    if all(all_checks):
        print("\n✅ ALL CHECKS PASSED!")
        print("\nProject is ready for use. Next steps:")
        print("  1. Run 'make eda' for exploratory data analysis")
        print("  2. Run 'make cv' for cross-validation")
        print("  3. Run 'make tune' for hyperparameter tuning")
        print("  4. Run 'make submit' to generate submission")
        print("\nOr run 'make all' to execute the complete pipeline.")
        return 0
    else:
        print("\n⚠ SOME CHECKS FAILED")
        print("\nPlease review the errors above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

