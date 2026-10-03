#!/usr/bin/env python3
"""
generate_gold_queries.py
------------------------
Generate or inspect evaluation/gold_queries.csv.

If gold_queries.csv already exists, prints a summary.
If it does not exist, creates a starter set of 100 aviation
safety queries covering the UATIE-EDM evaluation dimensions.

Place this file in: evaluation/generate_gold_queries.py
Run  : python evaluation/generate_gold_queries.py
Output: evaluation/gold_queries.csv
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR     = PROJECT_ROOT / "evaluation"
OUT_FILE     = EVAL_DIR / "gold_queries.csv"

STARTER_QUERIES = [
    # ── Runway / approach events ──────────────────────────────────────
    ("Q001", "late runway change during approach"),
    ("Q002", "runway was changed on final"),
    ("Q003", "ATC assigned different landing runway during arrival"),
    ("Q004", "new runway assigned close to touchdown"),
    ("Q005", "go-around due to late runway assignment"),
    ("Q006", "pilot refused runway change on short final"),
    ("Q007", "runway incursion during landing roll"),
    ("Q008", "wrong runway landing incident"),
    ("Q009", "crew landed on taxiway instead of runway"),
    ("Q010", "intersection takeoff insufficient runway length"),
    # ── Altitude / CFIT ───────────────────────────────────────────────
    ("Q011", "controlled flight into terrain mountainous area"),
    ("Q012", "premature descent below minimum descent altitude"),
    ("Q013", "altitude deviation during instrument approach"),
    ("Q014", "crew descended below cleared altitude"),
    ("Q015", "GPWS warning ignored by flight crew"),
    ("Q016", "terrain warning activation during final approach"),
    ("Q017", "aircraft hit obstacle during departure climb"),
    ("Q018", "crew busted altitude clearance en route"),
    ("Q019", "step down fix altitude non compliance"),
    ("Q020", "minimum safe altitude warning triggered"),
    # ── ATC phraseology / communication ──────────────────────────────
    ("Q021", "readback error ATC clearance misunderstood"),
    ("Q022", "crew read back wrong squawk code"),
    ("Q023", "frequency change missed by crew"),
    ("Q024", "misunderstood taxi clearance hot spot"),
    ("Q025", "ATC callsign confusion similar aircraft"),
    ("Q026", "pilot acknowledged wrong aircraft clearance"),
    ("Q027", "ATC instructed wrong aircraft to descend"),
    ("Q028", "crew missed altitude assignment in clearance"),
    ("Q029", "non-standard phraseology caused confusion"),
    ("Q030", "ambiguous ATC instruction led to deviation"),
    # ── Separation / TCAS ─────────────────────────────────────────────
    ("Q031", "TCAS resolution advisory both aircraft climbed"),
    ("Q032", "loss of separation below radar coverage"),
    ("Q033", "near mid-air collision VFR traffic"),
    ("Q034", "ATC lost track of aircraft on radar"),
    ("Q035", "wake turbulence encounter heavy departure"),
    ("Q036", "reduced separation minimum approval error"),
    ("Q037", "TCAS RA opposite to ATC instruction"),
    ("Q038", "aircraft entered controlled airspace without clearance"),
    ("Q039", "two aircraft converging final approach"),
    ("Q040", "ground proximity warning simultaneous go-around"),
    # ── Weather ───────────────────────────────────────────────────────
    ("Q041", "microburst encounter during approach"),
    ("Q042", "windshear warning on final approach"),
    ("Q043", "icing conditions airspeed anomaly"),
    ("Q044", "thunderstorm deviation fuel shortage"),
    ("Q045", "crew continued into IMC without clearance"),
    ("Q046", "freezing rain landing runway contamination"),
    ("Q047", "turbulence injury cabin crew not seated"),
    ("Q048", "severe clear air turbulence at cruise"),
    ("Q049", "low visibility taxi without follow-me"),
    ("Q050", "volcanic ash avoidance engine surge"),
    # ── Fuel / systems ────────────────────────────────────────────────
    ("Q051", "fuel exhaustion emergency declaration"),
    ("Q052", "crew declared emergency fuel minimum"),
    ("Q053", "hydraulic failure landing gear not extending"),
    ("Q054", "engine fire warning during climb"),
    ("Q055", "autopilot disconnect unexpected at low altitude"),
    ("Q056", "pitot static blockage erroneous airspeed"),
    ("Q057", "pressurisation failure emergency descent"),
    ("Q058", "bird strike engine ingestion at takeoff"),
    ("Q059", "brake failure rejected takeoff high speed"),
    ("Q060", "thrust reverser uncommanded deployment"),
    # ── Crew resource management / human factors ──────────────────────
    ("Q061", "crew incapacitation single pilot approach"),
    ("Q062", "fatigue reported by crew after long duty"),
    ("Q063", "captain overruled first officer correct callout"),
    ("Q064", "sterile cockpit violation below 10000 feet"),
    ("Q065", "distraction caused missed altitude capture"),
    ("Q066", "checklist not completed before takeoff"),
    ("Q067", "crew did not brief approach properly"),
    ("Q068", "automation surprise mode confusion"),
    ("Q069", "inappropriate use of autopilot during go-around"),
    ("Q070", "situational awareness loss night VMC"),
    # ── Ground / taxiway ──────────────────────────────────────────────
    ("Q071", "taxiway incursion wrong turn at night"),
    ("Q072", "aircraft crossed active runway without clearance"),
    ("Q073", "ground vehicle crossed runway behind landing aircraft"),
    ("Q074", "pushback conflict with taxiing aircraft"),
    ("Q075", "aircraft stopped on runway after landing missed exit"),
    ("Q076", "gate conflict push back rejected"),
    ("Q077", "wing tip strike during gate pushback"),
    ("Q078", "tow bar failure during towing incident"),
    ("Q079", "deicing incomplete crew unaware contamination"),
    ("Q080", "FOD on runway ingested by engine"),
    # ── Airspace / navigation ─────────────────────────────────────────
    ("Q081", "wrong waypoint entered flight management system"),
    ("Q082", "crew continued VOR approach without station passage"),
    ("Q083", "GPS signal lost oceanic track deviation"),
    ("Q084", "RNAV approach loaded incorrect procedure"),
    ("Q085", "navigational error wrong SID selected"),
    ("Q086", "transition level confusion altimeter setting error"),
    ("Q087", "crew entered incorrect QNH in descent"),
    ("Q088", "aircraft flew through restricted airspace"),
    ("Q089", "airspace infringement after ATC frequency loss"),
    ("Q090", "heading confusion magnetic variation high latitude"),
    # ── Emergency / abnormal ──────────────────────────────────────────
    ("Q091", "emergency evacuation slide deployed at gate"),
    ("Q092", "smoke in cockpit divert alternate airport"),
    ("Q093", "medical emergency passenger requested priority landing"),
    ("Q094", "bomb threat divert precautionary landing"),
    ("Q095", "door warning light during takeoff roll"),
    ("Q096", "cargo fire warning in flight diversion"),
    ("Q097", "crew squawked 7700 ATC response delayed"),
    ("Q098", "emergency frequency blocked no response ATC"),
    ("Q099", "stall warning activation at low altitude"),
    ("Q100", "go-around all engines windshear escape manoeuvre"),
]


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        df = pd.read_csv(OUT_FILE)
        print(f"[INFO] gold_queries.csv already exists ({len(df)} rows). Nothing changed.")
        print(df.head(10).to_string(index=False))
        return

    df = pd.DataFrame(STARTER_QUERIES, columns=["query_id", "query_text"])
    df.to_csv(OUT_FILE, index=False)
    print(f"[DONE] Created {OUT_FILE} with {len(df)} queries.")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
