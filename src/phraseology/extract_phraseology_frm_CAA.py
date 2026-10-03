# import camelot
# import pandas as pd
# from pathlib import Path

# pdf_path = "data/raw/CAA_ATC_Section8.pdf"   # your phraseology PDF
# out_csv  = "data/processed/CAA_ATC_section8_phraseology.csv"

# Path("data/processed").mkdir(parents=True, exist_ok=True)

# # Limit to the phraseology pages if you know the range, e.g. "10-30"
# tables = camelot.read_pdf(
#     pdf_path,
#     pages="all",
#     flavor="lattice"   # works best with visible grid lines like your sample
# )

# frames = []
# for t in tables:
#     df = t.df

#     # Your sample has exactly 2 logical columns
#     if df.shape[1] >= 2:
#         df.columns = ["circumstance", "phraseology"] + [
#             f"extra_{i}" for i in range(df.shape[1] - 2)
#         ]
#         frames.append(df[["circumstance", "phraseology"]])

# if not frames:
#     raise RuntimeError("No usable tables found; try flavor='stream' or adjust pages")

# full = pd.concat(frames, ignore_index=True)

# # Clean up: drop header repeats and empty lines
# full["circumstance"] = full["circumstance"].astype(str).str.strip()
# full["phraseology"]  = full["phraseology"].astype(str).str.strip()

# mask_header = (full["circumstance"].str.lower() == "circumstances") | \
#               (full["phraseology"].str.lower() == "phraseologies")
# full = full[~mask_header]
# full = full[(full["circumstance"] != "") | (full["phraseology"] != "")]

# # Add metadata
# full.insert(0, "source_doc", "CAA_ATC_Section8")

# full.to_csv(out_csv, index=False, encoding="utf-8")
# print(f"Wrote {len(full)} rows → {out_csv}")

# import re
# from pathlib import Path
# import pandas as pd
# import tabula

# pdf_path = "data/raw/CAA_ATC_Section8.pdf"   # change this
# out_csv = "data/processed/CAA_ATC_section8_phraseology.csv"

# Path("data/processed").mkdir(parents=True, exist_ok=True)

# # Read all tables from all pages.
# # If you know the phraseology page range, replace pages="all" with something like "10-25"
# tables = tabula.read_pdf(
#     pdf_path,
#     pages="all",
#     multiple_tables=True,
#     lattice=True,   # good for visible table borders like your sample
#     pandas_options={"dtype": str}
# )

# print(f"Tables found: {len(tables)}")

# clean_frames = []

# for idx, df in enumerate(tables):
#     if df is None or df.empty:
#         continue

#     # Normalize column names
#     df = df.copy()
#     df.columns = [str(c).strip() for c in df.columns]

#     # Force string type and clean whitespace
#     for col in df.columns:
#         df[col] = df[col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

#     # Keep tables with at least 2 columns
#     if df.shape[1] < 2:
#         continue

#     # Use first two columns as circumstance / phraseology
#     df2 = df.iloc[:, :2].copy()
#     df2.columns = ["circumstance", "phraseology"]

#     # Drop repeated table headers
#     mask_header = (
#         df2["circumstance"].str.lower().isin(["circumstances", "circumstance"]) |
#         df2["phraseology"].str.lower().isin(["phraseologies", "phraseology"])
#     )
#     df2 = df2[~mask_header]

#     # Drop obvious footer/header noise
#     def is_noise(text: str) -> bool:
#         t = text.strip()
#         if not t:
#             return False
#         if re.fullmatch(r"page \d+ of \d+", t.lower()):
#             return True
#         if re.fullmatch(r"\d{1,2} \w+ \d{4}", t):  # e.g. 8 February 2013
#             return True
#         if "CAA Standards & Procedures" in t:
#             return True
#         return False

#     df2 = df2[
#         ~df2["circumstance"].apply(is_noise) &
#         ~df2["phraseology"].apply(is_noise)
#     ]

#     # Remove rows where both sides are empty
#     df2 = df2[
#         (df2["circumstance"].str.strip() != "") |
#         (df2["phraseology"].str.strip() != "")
#     ]

#     if not df2.empty:
#         df2.insert(0, "source_doc", "CAA_ATC_Section8")
#         df2.insert(1, "table_index", idx)
#         clean_frames.append(df2)

# if not clean_frames:
#     raise RuntimeError("No usable phraseology tables were extracted")

# full = pd.concat(clean_frames, ignore_index=True)

# # Final tidy
# full["circumstance"] = full["circumstance"].str.replace(r"\s+", " ", regex=True).str.strip()
# full["phraseology"] = full["phraseology"].str.replace(r"\s+", " ", regex=True).str.strip()

# full.to_csv(out_csv, index=False, encoding="utf-8")
# print(f"Wrote {len(full)} rows to {out_csv}")

# import fitz  # pymupdf
# import csv
# from pathlib import Path

# pdf_path = "data/raw/CAA_ATC_Section8.pdf"
# out_csv  = "data/processed/CAA_ATC_section8_phraseology_pymupdf.csv"

# Path("data/processed").mkdir(parents=True, exist_ok=True)

# doc = fitz.open(pdf_path)

# rows = []

# # Adjust page range to where phraseology tables live (0-based indexes)
# start_page = 0
# end_page   = len(doc)   # or e.g. 10, 20 etc.

# for pno in range(start_page, end_page):
#     page = doc[pno]
#     blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
#     if not blocks:
#         continue

#     # Find approximate vertical split between left/right columns
#     xs = [b[0] for b in blocks] + [b[2] for b in blocks]
#     x_min, x_max = min(xs), max(xs)
#     mid_x = (x_min + x_max) / 2

#     left_lines = []
#     right_lines = []

#     for (x0, y0, x1, y1, text, *_ ) in blocks:
#         t = text.strip()
#         if not t:
#             continue

#         # Skip page footer/header noise
#         if "CAA Standards & Procedures" in t:
#             continue
#         if "Section 8" in t and "Page" in t:
#             continue
#         if t.lower() in ("circumstances", "phraseologies", "general phraseology"):
#             continue

#         # Rough left/right split by x0
#         if x1 <= mid_x:          # fully left
#             left_lines.append(t)
#         elif x0 >= mid_x:        # fully right
#             right_lines.append(t)
#         else:
#             # Overlapping – assign based on center
#             cx = (x0 + x1) / 2
#             (left_lines if cx < mid_x else right_lines).append(t)

#     # Join lines into bigger blocks. This is heuristic and may need tuning.
#     left_text  = " ".join(left_lines)
#     right_text = " ".join(right_lines)

#     # Very simple split: assume multiple rows separated by lowercase→uppercase patterns etc.
#     # For now, treat entire page as one block if both sides non-empty.
#     if left_text and right_text:
#         rows.append({
#             "source_doc": "CAA_ATC_Section8",
#             "page": pno + 1,
#             "circumstance": left_text,
#             "phraseology": right_text,
#         })

# doc.close()

# fieldnames = ["source_doc", "page", "circumstance", "phraseology"]
# with open(out_csv, "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     for r in rows:
#         writer.writerow(r)

# print(f"Wrote {len(rows)} page-level rows → {out_csv}")


# import tabula
# import pandas as pd

# pdf_path = "data/raw/CAA_ATC_Section8.pdf"  # adjust

# # Try only the phraseology page you showed (human page 12 -> pages="12")
# tables = tabula.read_pdf(
#     pdf_path,
#     pages="12",
#     multiple_tables=True,
#     lattice=True
# )

# print("tables found:", len(tables))
# for i, df in enumerate(tables):
#     print(f"\n--- TABLE {i} ---")
#     print(df.head(10))                      


# import re
# from pathlib import Path
# import pandas as pd
# import tabula

# pdf_path = "data/raw/CAA_ATC_Section8.pdf"   # change this
# out_csv = "data/processed/CAA_ATC_section8_phraseology.csv"

# Path("data/processed").mkdir(parents=True, exist_ok=True)

# # Read all tables from all pages.
# # If you know the phraseology page range, replace pages="all" with something like "10-25"
# tables = tabula.read_pdf(
#     pdf_path,
#     pages="all",
#     multiple_tables=True,
#     lattice=True,   # good for visible table borders like your sample
#     pandas_options={"dtype": str}
# )

# print(f"Tables found: {len(tables)}")

# clean_frames = []

# for idx, df in enumerate(tables):
#     if df is None or df.empty:
#         continue

#     # Normalize column names
#     df = df.copy()
#     df.columns = [str(c).strip() for c in df.columns]

#     # Force string type and clean whitespace
#     for col in df.columns:
#         df[col] = df[col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

#     # Keep tables with at least 2 columns
#     if df.shape[1] < 2:
#         continue

#     # Use first two columns as circumstance / phraseology
#     df2 = df.iloc[:, :2].copy()
#     df2.columns = ["circumstance", "phraseology"]

#     # Drop repeated table headers
#     mask_header = (
#         df2["circumstance"].str.lower().isin(["circumstances", "circumstance"]) |
#         df2["phraseology"].str.lower().isin(["phraseologies", "phraseology"])
#     )
#     df2 = df2[~mask_header]

#     # Drop obvious footer/header noise
#     def is_noise(text: str) -> bool:
#         t = text.strip()
#         if not t:
#             return False
#         if re.fullmatch(r"page \d+ of \d+", t.lower()):
#             return True
#         if re.fullmatch(r"\d{1,2} \w+ \d{4}", t):  # e.g. 8 February 2013
#             return True
#         if "CAA Standards & Procedures" in t:
#             return True
#         return False

#     df2 = df2[
#         ~df2["circumstance"].apply(is_noise) &
#         ~df2["phraseology"].apply(is_noise)
#     ]

#     # Remove rows where both sides are empty
#     df2 = df2[
#         (df2["circumstance"].str.strip() != "") |
#         (df2["phraseology"].str.strip() != "")
#     ]

#     if not df2.empty:
#         df2.insert(0, "source_doc", "CAA_ATC_Section8")
#         df2.insert(1, "table_index", idx)
#         clean_frames.append(df2)

# if not clean_frames:
#     raise RuntimeError("No usable phraseology tables were extracted")

# full = pd.concat(clean_frames, ignore_index=True)

# # Final tidy
# full["circumstance"] = full["circumstance"].str.replace(r"\s+", " ", regex=True).str.strip()
# full["phraseology"] = full["phraseology"].str.replace(r"\s+", " ", regex=True).str.strip()

# full.to_csv(out_csv, index=False, encoding="utf-8")
# print(f"Wrote {len(full)} rows to {out_csv}")


# import re
# from pathlib import Path
# import pandas as pd
# import tabula

# pdf_path = "data/raw/CAA_ATC_Section8.pdf"    # adjust
# out_csv  = "data/processed/CAA_ATC_section8_phraseology.csv"

# Path("data/processed").mkdir(parents=True, exist_ok=True)

# # Adjust the page range to all pages that contain phraseology tables
# tables = tabula.read_pdf(
#     pdf_path,
#     pages="12-20",         # example range; tweak as needed
#     multiple_tables=True,
#     lattice=True
# )

# frames = []

# for df in tables:
#     if df is None or df.empty:
#         continue

#     df = df.copy()
#     df.columns = [str(c).strip() for c in df.columns]

#     # Case A: already proper Circumstances / Phraseologies columns
#     if df.shape[1] >= 2 and "Circumstances" in df.columns[0] and "Phraseolog" in df.columns[1]:
#         df2 = df.iloc[:, :2].copy()
#         df2.columns = ["circumstance", "phraseology"]

#     # Case B: header row is the first row (like your TABLE 1)
#     elif df.shape[1] >= 2 and "Circumstances" in str(df.iloc[0,0]) and "Phraseolog" in str(df.iloc[0,1]):
#         df2 = df.iloc[1:, :2].copy()
#         df2.columns = ["circumstance", "phraseology"]

#     else:
#         continue

#     # Clean whitespace and line breaks
#     for col in ["circumstance", "phraseology"]:
#         df2[col] = (
#             df2[col].fillna("")
#             .astype(str)
#             .str.replace(r"\\r", " ", regex=True)
#             .str.replace(r"\s+", " ", regex=True)
#             .str.strip()
#         )

#     # Drop any header repeats or empty lines
#     mask_header = (df2["circumstance"].str.lower() == "circumstances") | \
#                   (df2["phraseology"].str.lower() == "phraseologies")
#     df2 = df2[~mask_header]
#     df2 = df2[(df2["circumstance"] != "") | (df2["phraseology"] != "")]

#     if not df2.empty:
#         frames.append(df2)

# if not frames:
#     raise RuntimeError("No phraseology tables detected in selected pages")

# full = pd.concat(frames, ignore_index=True)

# # Optional: trim leading bullet labels like "a)" from phraseology into a separate field later
# full.insert(0, "source_doc", "CAA_ATC_Section8")

# full.to_csv(out_csv, index=False, encoding="utf-8")
# print(f"Wrote {len(full)} rows → {out_csv}")


import pandas as pd
from pathlib import Path

in_csv  = "data/processed/CAA_ATC_section8_phraseology.csv"   # existing file
out_csv = "data/processed/CAA_ATC_section8_phraseology_merged.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_csv, dtype=str).fillna("")

merged_rows = []
current = None

for _, row in df.iterrows():
    circ = row["circumstance"].strip()
    phr  = row["phraseology"].strip()

    if circ and phr:
        if current:
            merged_rows.append(current)
        current = {
            "source_doc": row.get("source_doc", "CAA_ATC_Section8"),
            "circumstance": circ,
            "phraseology": phr,
        }
    elif not circ and phr and current:
        current["phraseology"] = (current["phraseology"] + " " + phr).strip()
    else:
        continue

if current:
    merged_rows.append(current)

out_df = pd.DataFrame(merged_rows)
out_df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"Wrote {len(out_df)} merged rows → {out_csv}")


import re
import pandas as pd
from pathlib import Path

in_csv  = "data/processed/CAA_ATC_section8_phraseology_merged.csv"
out_csv = "data/processed/CAA_ATC_section8_phraseology_rules.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_csv, dtype=str).fillna("")

rows_out = []

for _, row in df.iterrows():
    circ = row["circumstance"].strip()
    text = row["phraseology"].strip()

    if not text:
        continue

    # Insert a separator before each bullet marker (a), b), z), aa), etc.)
    s = re.sub(r"(\s)([a-z]{1,2}\))", r"\1||\2", text)
    parts = []
    for chunk in s.split("||"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^([a-z]{1,2})\)\s*(.*)", chunk, flags=re.I)
        if m:
            bullet_id = m.group(1)
            phrase = m.group(2).strip(" ;")
            parts.append((bullet_id, phrase))
        else:
            # continuation of previous bullet
            if parts:
                parts[-1] = (parts[-1][1], (parts[-1][1] + " " + chunk).strip(" ;"))

    if not parts:
        # no bullets detected, keep whole text as one rule
        rows_out.append({
            "source_doc": row.get("source_doc", "CAA_ATC_Section8"),
            "circumstance": circ,
            "bullet_id": "",
            "phraseology": text,
        })
    else:
        for bullet_id, phrase in parts:
            rows_out.append({
                "source_doc": row.get("source_doc", "CAA_ATC_Section8"),
                "circumstance": circ,
                "bullet_id": bullet_id,
                "phraseology": phrase,
            })

rules_df = pd.DataFrame(rows_out)
rules_df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"Wrote {len(rules_df)} phraseology rules → {out_csv}")