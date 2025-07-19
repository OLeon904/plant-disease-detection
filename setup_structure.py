import os

folders = [
    "data",
    "models",
    "src",
    "app",
    "results",
    "report"
]

files = {
    "src/train.py": "",
    "src/evaluate.py": "",
    "src/preprocess.py": "",
    "src/utils.py": "",
    "app/streamlit_app.py": "",
    "report/final_report.md": "",
    "README.md": ""
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Create files
for filepath, content in files.items():
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created file: {filepath}")
    else:
        print(f"File already exists: {filepath}")
