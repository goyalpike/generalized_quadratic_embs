from pathlib import Path

A = 100
B = 300
def hello():
    print("Hello world!")

    

# Get the directory of the current file
THIS_DIR = Path(__file__).resolve().parent.parent

# Get the parent directory of the  directory
PROJECT_DIR= Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

print(PROJECT_DIR)
print(DATA_DIR)
