from pathlib import Path
import re
import pandas as pd

# ---------------------------------------------------------
# 1. Input raw text files for both manuals
# ---------------------------------------------------------

sources = [
    ("Doc9432_example", Path("data/processed/doc9432_full_raw.txt")),
    ("Doc4444_standard", Path("data/processed/doc4444_full_raw.txt")),
]

rows = []


# ---------------------------------------------------------
# 2. Phase detection helper
# ---------------------------------------------------------

def update_phase_from_line(line: str, current_phase: str) -> str:
    """Infer flight phase or ground from headings/context."""
    text = line.upper()
    if "DEPARTURE" in text or "TAKE-OFF" in text or "TAKEOFF" in text:
        return "Departure"
    if "APPROACH" in text or "FINAL APPROACH" in text or "ILS" in text:
        return "Approach"
    if "LANDING" in text or "TOUCHDOWN" in text:
        return "Landing"
    if "EN-ROUTE" in text or "EN ROUTE" in text or "CRUISE" in text:
        return "Enroute"
    if "GROUND" in text or "TAXI" in text:
        return "Ground"
    return current_phase


# ---------------------------------------------------------
# 3. Main extraction loop
# ---------------------------------------------------------

for source_name, path in sources:
    if not path.exists():
        print(f"⚠️ Missing raw file for {source_name}: {path}")
        continue

    print(f"Processing {source_name} from {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")

    current_phase = "Unknown"
    current_role_hint = "Unknown"  # used only for Doc9432_example

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Update phase from headings
        current_phase = update_phase_from_line(line, current_phase)

        # ---------------- Doc 9432 rules ----------------
        if source_name == "Doc9432_example":
            # Speaker hints on separate lines
            if re.match(r"^ATC\b", line, re.IGNORECASE):
                current_role_hint = "ATC"
                continue
            if re.match(r"^PILOT\b", line, re.IGNORECASE):
                current_role_hint = "Pilot"
                continue

            # Rule 1: UPPERCASE command-like phrases
            if (
                line.isupper()
                and 8 <= len(line) <= 90
                and len(line.split()) <= 12
                and re.search(
                    r"\b(CLEARED|MAINTAIN|DESCEND|CLIMB|HOLD|CONTACT|REPORT|"
                    r"GO AROUND|LINE UP|TAXI|CROSS|REQUEST|CHECK|CONFIRM)\b",
                    line,
                )
            ):
                rows.append({
                    "source": source_name,
                    "phase": current_phase,
                    "role": current_role_hint,  # may be ATC/Pilot/Unknown
                    "standard_phrase": line,
                    "notes": "auto_uppercase_9432",
                })
                continue

            # Rule 2: inline ATC/PILOT: patterns
            m = re.match(r"^(ATC|PILOT)\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
            if m:
                role = m.group(1).upper()
                phrase = m.group(2).strip()
                if 3 < len(phrase) <= 80 and len(phrase.split()) <= 15:
                    current_role_hint = "ATC" if role == "ATC" else "Pilot"
                    rows.append({
                        "source": source_name,
                        "phase": current_phase,
                        "role": current_role_hint,
                        "standard_phrase": phrase,
                        "notes": "auto_role_prefix_9432",
                    })
                continue

        # ---------------- Doc 4444 rules ----------------
        elif source_name == "Doc4444_standard":
            # Only consider all-uppercase lines for now
            if not line.isupper():
                continue

            # Hard filter: line must start with typical call sign or RT verb
            if not re.match(
                r"^([A-Z0-9\-]+|RUNWAY|CLIMB|DESCEND|MAINTAIN|HOLD|CONTACT|"
                r"REPORT|REQUEST|TAXI|CROSS|CLEARED|GO AROUND)\b",
                line,
            ):
                continue

            # Length / word-count filters to remove long definition sentences
            if not (5 <= len(line) <= 80):
                continue
            if len(line.split()) > 12:
                continue

            # Must contain at least one radiotelephony verb
            if not re.search(
                r"\b(CLEARED|MAINTAIN|DESCEND|CLIMB|HOLD|CONTACT|REPORT|"
                r"GO AROUND|LINE UP|TAXI|CROSS|REQUEST)\b",
                line,
            ):
                continue

            rows.append({
                "source": source_name,
                "phase": current_phase,
                "role": "Unknown",  # Doc 4444 tables rarely tag ATC/Pilot explicitly
                "standard_phrase": line,
                "notes": "auto_uppercase_4444",
            })
            continue

# ---------------------------------------------------------
# 4. Build DataFrame and clean
# ---------------------------------------------------------

df = pd.DataFrame(rows)

if df.empty:
    print("❌ No candidate phrases found. Check file paths and rules.")
else:
    # Normalize whitespace in phrases
    df["standard_phrase"] = (
        df["standard_phrase"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove non-text or very short phrases
    df = df[df["standard_phrase"].str.contains(r"[A-Za-z]", na=False)]
    df = df[df["standard_phrase"].str.split().str.len() >= 2]

    # Normalize phase labels to controlled set
    phase_map = {
        "Ground": "Ground",
        "Departure": "Departure",
        "Approach": "Approach",
        "Landing": "Landing",
        "Enroute": "Enroute",
    }
    df["phase"] = df["phase"].map(phase_map).fillna("Unknown")

    # Normalize role
    role_map = {
        "ATC": "ATC",
        "Pilot": "Pilot",
        "Unknown": "Unknown",
    }
    df["role"] = df["role"].map(role_map).fillna("Unknown")

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["source", "phase", "role", "standard_phrase"])

    # Reorder columns for master schema
    df = df[["source", "phase", "role", "standard_phrase", "notes"]]

    out_path = Path("data/processed/icao_phraseology_master.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("✅ Master phraseology shape:", df.shape)
    print("✅ Saved to:", out_path)