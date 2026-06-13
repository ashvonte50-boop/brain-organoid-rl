"""Extract methods data, parameters, and important text files."""
import json, re, shutil
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/Admin/brain-organoid-rl")
OUTPUT_DIR = PROJECT_ROOT / "important_paper_materials_and_results"

with open(OUTPUT_DIR / "00_full_file_inventory.json", "r") as f:
    inventory = json.load(f)

methods_data = {}
important_txt = []

for txt_entry in inventory["txt_files"] + inventory["log_files"]:
    path = txt_entry["path"]
    fname = txt_entry["name"].lower()

    if txt_entry["size_kb"] > 500:
        continue

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        relevant_terms = ["gamma", "neuron", "stdp", "replay", "w_slow", "seed",
                         "retention", "r2", "alpha", "beta", "p-value", "result",
                         "mod2", "mod3", "mod4", "mod5", "law_fit", "summary",
                         "serial", "position", "consolidation", "cascade"]

        if not any(term in content.lower() for term in relevant_terms):
            continue

        entry = {
            "file": txt_entry["relative"],
            "size_kb": txt_entry["size_kb"],
            "preview": content[:2000],
            "full_content": content if txt_entry["size_kb"] < 50 else content[:10000]
        }

        if any(key in fname for key in ["summary", "law_fit", "param_ground",
                                         "mod2", "mod4", "mod5", "result",
                                         "replacement", "narrative", "finding",
                                         "report", "cheatsheet"]):
            entry["priority"] = "HIGH"
            important_txt.append(entry)

        methods_data[txt_entry["relative"]] = entry

    except Exception as e:
        pass

print(f"=== TXT/LOG EXTRACTION ===")
print(f"Relevant text files found: {len(methods_data)}")
print(f"High-priority text files: {len(important_txt)}")

for i, entry in enumerate(important_txt):
    safe_name = entry["file"].replace("/", "_").replace("\\", "_").replace(":", "_")
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    out_path = OUTPUT_DIR / "methods_extracted" / f"{i:02d}_{safe_name}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: {entry['file']}\n")
        f.write("=" * 60 + "\n")
        f.write(entry["full_content"])
    print(f"  Saved: {out_path.name}")

# Extract parameters from Python files
print("\n=== EXTRACTING PARAMETERS FROM PYTHON FILES ===")
params_found = {}
for py_entry in inventory["python_scripts"]:
    path = py_entry["path"]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        param_patterns = [
            (r"GAMMA\s*=\s*([\d.]+)", "GAMMA"),
            (r"N_NEURONS\s*=\s*(\d+)", "N_NEURONS"),
            (r"TAU_SLOW\s*=\s*([\d.]+)", "TAU_SLOW"),
            (r"W_MAX\s*=\s*([\d.]+)", "W_MAX"),
            (r"ETA\s*=\s*([\d.]+)", "ETA"),
            (r"A_PLUS\s*=\s*([\d.]+)", "A_PLUS"),
            (r"A_MINUS\s*=\s*([\d.]+)", "A_MINUS"),
            (r"REPLAY_COHERENCE_THR\s*=\s*([\d.]+)", "REPLAY_COHERENCE_THR"),
            (r"STDP_GATE_BIAS\s*=\s*([\d.]+)", "STDP_GATE_BIAS"),
            (r"MB_BOOST\s*=\s*([\d.]+)", "MB_BOOST"),
            (r"N_MEMORIES\s*=\s*(\d+)", "N_MEMORIES"),
            (r"CORE_SIZE\s*=\s*(\d+)", "CORE_SIZE"),
            (r"N_EXC\s*=\s*(\d+)", "N_EXC"),
            (r"N_INH\s*=\s*(\d+)", "N_INH"),
            (r"N_MODULES\s*=\s*(\d+)", "N_MODULES"),
            (r"INTRA_P\s*=\s*([\d.]+)", "INTRA_P"),
            (r"INTER_P\s*=\s*([\d.]+)", "INTER_P"),
            (r"N_PRESENTATIONS\s*=\s*(\d+)", "N_PRESENTATIONS"),
            (r"N_TRIALS\s*=\s*(\d+)", "N_TRIALS"),
            (r"DT\s*=\s*([\d.]+)", "DT"),
            (r"ENCODING_DURATION\s*=\s*([\d.]+)", "ENCODING_DURATION"),
            (r"REST_DURATION\s*=\s*([\d.]+)", "REST_DURATION"),
            (r"PROBE_DURATION\s*=\s*([\d.]+)", "PROBE_DURATION"),
        ]

        for pattern, name in param_patterns:
            matches = re.findall(pattern, content)
            if matches:
                params_found[name] = params_found.get(name, set())
                for m in matches[:3]:
                    params_found[name].add(m)

    except:
        pass

params_clean = {k: sorted(list(v)) for k, v in params_found.items()}
with open(OUTPUT_DIR / "methods_extracted" / "03_parameters_extracted.json", "w") as f:
    json.dump(params_clean, f, indent=2)
print("Parameters extracted:")
for k, v in sorted(params_clean.items()):
    print(f"  {k}: {v}")

# Also look for summary files in results directories
print("\n=== COPYING SUMMARY FILES ===")
for txt in Path(PROJECT_ROOT).rglob("*summary*.txt"):
    if "important_paper" in str(txt):
        continue
    if "mod" in str(txt).lower() or "result" in str(txt).lower() or "major" in str(txt).lower():
        try:
            dst = OUTPUT_DIR / "methods_extracted" / f"SUMMARY_{txt.parent.name}_{txt.name}"
            shutil.copy2(txt, dst)
            print(f"  Copied summary: {txt.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"  Failed: {txt.name}: {e}")

for txt in Path(PROJECT_ROOT).rglob("*summary*.csv"):
    if "important_paper" in str(txt):
        continue
    try:
        dst = OUTPUT_DIR / "methods_extracted" / f"SUMMARY_{txt.parent.name}_{txt.name}"
        shutil.copy2(txt, dst)
        print(f"  Copied summary CSV: {txt.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        pass
