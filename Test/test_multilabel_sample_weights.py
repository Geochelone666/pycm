import numpy as np
import pytest

from pycm import MultiLabelCM


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("classes", [["cat", "dog"], ["cat", "dog", "bird"]])
def test_sample_confusion_matrix_uses_current_sample_weight(classes, as_array):
    weights = np.array([2, 5]) if as_array else [2, 5]
    actual = [{"dog"}, {"cat", "dog"}]
    predicted = [{"cat"}, {"cat"}]
    weighted = MultiLabelCM(actual, predicted, sample_weight=weights, classes=classes)
    unweighted = MultiLabelCM(actual, predicted, classes=classes)

    for index, weight in enumerate(weights):
        expected = unweighted.get_cm_by_sample(index)
        result = weighted.get_cm_by_sample(index)
        assert result.table == {
            row: {col: count * weight for col, count in entries.items()}
            for row, entries in expected.table.items()
        }
        assert list(result.weights) == [weight] * len(classes)
        assert weighted.get_cm_by_sample(index) is result

    np.testing.assert_array_equal(weights, [2, 5])
