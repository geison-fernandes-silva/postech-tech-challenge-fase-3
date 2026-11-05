from medical_triage.data import make_dataset


def test_synthetic_dataset_has_minimum_required_volume_and_labels():
    dataset = make_dataset()
    assert len(dataset) >= 2_000
    assert set(dataset["urgency"]) == {"normal", "atencao", "urgente"}
    assert dataset["report_text"].str.len().min() > 10
