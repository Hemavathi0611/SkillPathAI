import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

datasets = {
    "CAREERPATH AI": DATA_DIR / "career_dataset.csv",
    "JOBMATCH AI": DATA_DIR / "job_dataset.csv"
}

for name, file_path in datasets.items():
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp1252")

    print("\n1. DATASET SHAPE")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n3. DATA TYPES")
    print(df.dtypes.to_string())

    print("\n4. MISSING VALUES")
    print(df.isnull().sum().to_string())

    print("\n5. DUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\n6. FIRST FIVE RECORDS")
    print(df.head().to_string(index=False))

    print("\n7. DATASET SUMMARY")
    print(df.describe(include="all").transpose().to_string())

print("\nINSPECTION COMPLETED")
