def recommend(label: str, severe: bool) -> str:
    prefix = "Immediate action: " if severe else "Recommendation: "
    text = {
        "sharp_corner_edge_zone": "add corner guards or padding and keep children away from this edge-heavy area.",
        "wet_floor_or_spill": "wipe the spill, mark the area, and improve lighting to reduce slip risk.",
        "obstacle_in_path": "remove or relocate the object from the walking path.",
        "fall_risk_person_on_floor": "check the person immediately and keep the path clear.",
        "recurring_hazard_zone": "review this repeated hazard zone and rearrange furniture or routines.",
    }.get(label, "review this safety risk.")
    return prefix + text
