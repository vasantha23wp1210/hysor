import os
import pandas as pd
from owlready2 import *

# ---------- Paths ----------
OWL_INPUT = "src/ontology/uatie_edm_ontology.owl"
OWL_OUTPUT = "src/ontology/uatie_fused.owl"
CSV_NTSB = "data/raw/NTSB_database.csv"
CSV_AIRLINE = "data/raw/airline_safety.csv"
CSV_CRASHES = "data/raw/plane_crashes.csv"
CSV_FUSED = "data/processed/fused_ntsb_airline_crashes.csv"


def load_ontology():
    if not os.path.exists(OWL_INPUT):
        print(f"❌ OWL missing: {OWL_INPUT}")
        print("Run create_base_owl.py first!")
        raise SystemExit(1)

    onto = get_ontology(f"file://{os.path.abspath(OWL_INPUT)}").load()
    print(f"✅ OWL loaded: {len(list(onto.classes()))} classes")

    classes = list(onto.classes())
    print("All classes in ontology:")
    for c in classes:
        print(" ", c.name)

    incident_class = classes[0]  # temporary generic incident holder
    print("Using incident_class =", incident_class)

    return onto, incident_class


def load_csvs():
    csv_files = [CSV_NTSB, CSV_AIRLINE, CSV_CRASHES]
    csv_files = [f for f in csv_files if os.path.exists(f)]

    if not csv_files:
        print("❌ No CSVs found in data/raw/. Run download_data.py and add NTSB manually.")
        raise SystemExit(1)

    print(f"🔍 {len(csv_files)} CSVs found")
    for f in csv_files:
        print(" ", f)

    df_list = []
    for f in csv_files:
        df = pd.read_csv(f, nrows=2000, encoding="utf-8", low_memory=False)
        df["source"] = os.path.basename(f)
        df_list.append(df)
        print(f"  {len(df)} rows: {f}")

    master_df = pd.concat(df_list, ignore_index=True)
    print(f"\n🎉 {len(master_df)} records loaded!")
    return master_df


def choose_text_columns(master_df: pd.DataFrame):
    narrative_cols = [
        c
        for c in master_df.columns
        if any(k in c.lower() for k in ["narrative", "summary", "report"])
    ]
    if not narrative_cols:
        narrative_cols = [c for c in master_df.columns if master_df[c].dtype == "object"]

    print("Using text columns:", narrative_cols)
    return narrative_cols


def fuse_to_owl(onto, incident_class, master_df, narrative_cols):
    triples = 0
    with onto:
        for idx, row in master_df.iterrows():
            text = " ".join(str(row.get(c, "")) for c in narrative_cols).lower()
            if any(
                word in text
                for word in ["readback", "clear", "land", "error", "controller"]
            ):
                inst = incident_class(f"incident_{idx}")
                inst.comment = [str(text)[:300]]
                triples += 1

    print(f"\n✅ FUSED {triples} incidents to OWL!")
    return triples


def main():
    onto, incident_class = load_ontology()
    master_df = load_csvs()
    narrative_cols = choose_text_columns(master_df)

    os.makedirs("data/processed", exist_ok=True)
    fuse_to_owl(onto, incident_class, master_df, narrative_cols)

    onto.save(file=OWL_OUTPUT, format="rdfxml")
    master_df.to_csv(CSV_FUSED, index=False)

    print("Files saved:")
    print("  -", OWL_OUTPUT)
    print("  -", CSV_FUSED)
    print("\nNext: run RAG demo over fused CSV.")


    fused = pd.read_csv("data/processed/fused_ntsb_airline_crashes.csv")
    asrs  = pd.read_csv("data/processed/asrs_for_fusion.csv")

    # Make sure column names line up
    common_cols = sorted(set(fused.columns) & set(asrs.columns))
    print("Common columns:", common_cols)

    # Restrict ASRS to those columns, then align order with fused
    asrs_aligned = asrs[common_cols]
    fused_aligned = fused[common_cols]

    fused_with_asrs = pd.concat([fused_aligned, asrs_aligned], ignore_index=True)
    print(fused_with_asrs.shape)

    fused_with_asrs.to_csv("data/processed/fused_with_asrs.csv", index=False)

if __name__ == "__main__":
    main()