import pandas as pd

raw = pd.read_csv("data/raw/ASRS-raw-dataset-aviation-safety.csv")
clean = pd.read_csv("data/raw/ASRS-clean-dataset-aviation-safety.csv")

print(raw.shape, clean.shape)
print(raw.columns)
print(clean.columns)

keep_cols = [
    "event_id", "event_datetime",
    "location_airport", "location_state",
    "aircraft1_operator", "aircraft1_model", "aircraft1_flight_phase", "aircraft1_airspace",
    "person1_role", "person1_human_factors", "person1_communication_breakdown",
    "event_anomaly", "event_detection_phase", "event_outcome", "primary_problem",
    "report1_narrative", "report2_narrative", "report_summary",
]


asrs_core = clean[keep_cols].copy()
print(asrs_core.shape)
asrs_core.to_csv("data/processed/asrs_core.csv", index=False)


asrs_core = pd.read_csv("data/processed/asrs_core.csv")

asrs_for_fusion = pd.DataFrame({
    "source": "ASRS",
    "event_id": asrs_core["event_id"],
    "datetime": asrs_core["event_datetime"],
    "operator": asrs_core["aircraft1_operator"],
    "location": asrs_core["location_airport"],
    "phase_of_flight": asrs_core["aircraft1_flight_phase"],
    "primary_problem": asrs_core["primary_problem"],
    "summary": asrs_core["report_summary"],
})

asrs_for_fusion.to_csv("data/processed/asrs_for_fusion.csv", index=False)
print(asrs_for_fusion.head(3))