from pathlib import Path

folder_path = Path("C:/Users/misha/Desktop/petnica projekat 2026/leto/FedPapersCorpus")

# Get full Path objects
full_paths = [
    f.resolve() for f in folder_path.rglob('*') if f.is_file()
]

spec_path = Path("C:/Users/misha/Desktop/petnica projekat 2026/leto/FedPapersCorpus/jay")

reci = {}

nt=0
for path in full_paths:
    if not path.is_relative_to(spec_path):
        # print(path)
        continue

    with open(path, 'r') as f:
        text = f.read()
        words = text.split()
        n = len(words)
        # print(n)
        nt += n
        continue
        for Rec in words:
            rec = ""
            for c in Rec:
                if c not in (",", ".", "?", "!", ";", ":"): rec+=c.lower()

            if rec in reci:
                reci[rec] += 1
            else:
                reci[rec] = 1

# N = 100
# topN = [k for k, v in sorted(reci.items(), key=lambda item: item[1], reverse=True)][:N]
# print(topN)
            
print("total", nt)