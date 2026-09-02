from pathlib import Path

folder_path = Path("C:/Users/misha/Desktop/petnica projekat 2026/leto/FedPapersCorpus")

# Get full Path objects
full_paths = [
    f.resolve() for f in folder_path.rglob('*') if f.is_file()
]

for path in full_paths:
    with open(path, 'r') as f:
        text = f.readlines()

    n = 0
    for line in text:
        words = line.split()
        n += len(words)

    print(n)