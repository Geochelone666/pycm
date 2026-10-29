"""Saved vector objects retain the explicit class domain and ordering."""
import json
import warnings
import pytest
from pycm import ConfusionMatrix


@pytest.mark.parametrize("classes", [[1, 0], [2, 1, 0], [0, 1, 2]])
@pytest.mark.parametrize("weights", [None, [2, 5, 3]])
@pytest.mark.parametrize("string_labels", [False, True])
@pytest.mark.parametrize("save_vector", [True, False])
def test_save_load_preserves_explicit_classes(tmp_path, classes, weights, save_vector, string_labels):
    actual_values, predicted_values = [0, 1, 1], [0, 1, 0]
    if string_labels:
        classes = list(map(str, classes))
        actual_values = list(map(str, actual_values))
        predicted_values = list(map(str, predicted_values))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = ConfusionMatrix(actual_values, predicted_values, classes=classes, sample_weight=weights)
    name = str(tmp_path / "matrix")
    assert expected.save_obj(name, save_vector=save_vector)["Status"]
    with open(name + ".obj") as file:
        actual = ConfusionMatrix(file=file)
    assert actual.classes == expected.classes
    assert actual.table == expected.table
    assert actual.overall_stat == expected.overall_stat
    assert actual.class_stat == expected.class_stat
    if save_vector:
        assert actual.weights == expected.weights


def test_legacy_vector_only_json_still_loads(tmp_path):
    name = tmp_path / "legacy.obj"
    name.write_text(json.dumps({"Actual-Vector": [0, 1, 1],
                                "Predict-Vector": [0, 1, 0], "Digit": 5}))
    with name.open() as file:
        actual = ConfusionMatrix(file=file)
    assert actual.table == ConfusionMatrix([0, 1, 1], [0, 1, 0]).table
