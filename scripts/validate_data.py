import csv, collections, pathlib
D = pathlib.Path("data")
def load(name):
    with open(D / name, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f); rows = list(r)
    return r.fieldnames, rows
for n in ["answers.csv", "brand_facts.csv", "development_gold.csv"]:
    cols, rows = load(n)
    print(f"\n== {n}: {len(rows)} rows")
    for c in cols:
        vals = [r[c] for r in rows]
        t = 'int' if all(v.lstrip('-').isdigit() for v in vals if v) else 'str'
        print(f"  {c!r}: empty={sum(v == '' for v in vals)} distinct={len(set(vals))} type={t}")
        if len(set(vals)) <= 12:
            print("    values:", dict(collections.Counter(vals)))
a_cols, a = load("answers.csv"); _, b = load("brand_facts.csv"); _, g = load("development_gold.csv")
print("\nCHECK answers==50:", len(a) == 50)
print("CHECK brand_facts==132:", len(b) == 132)
print("CHECK gold==376:", len(g) == 376)
ids = [r[a_cols[0]] for r in a]
print("IDs:", ids[0], "...", ids[-1], "| unique:", len(set(ids)) == len(ids))
print("A01-A40 present:", all(f"A{i:02d}" in ids for i in range(1, 41)))
print("A41-A50 present:", all(f"A{i:02d}" in ids for i in range(41, 51)))
