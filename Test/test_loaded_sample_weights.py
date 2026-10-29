"""Saved vector matrices must have the same checks as constructor inputs."""
import json
import tempfile
import pytest
from pycm import ConfusionMatrix
from pycm.errors import pycmVectorError


def load(weights, actual=None, predicted=None):
    data={'Actual-Vector': [0,1,1] if actual is None else actual,
          'Predict-Vector': [0,1,0] if predicted is None else predicted,
          'Digit':5}
    if weights != 'missing': data['Sample-Weight']=weights
    with tempfile.TemporaryFile(mode='w+') as file:
        json.dump(data,file);file.seek(0)
        return ConfusionMatrix(file=file)


@pytest.mark.parametrize('weights',[[2],[-1,2,3],[1,float('nan'),3],[1,float('inf'),3],
                                   [1,'2',3],[1,None,3]])
def test_saved_invalid_weights_rejected(weights):
    with pytest.raises(pycmVectorError): load(weights)


@pytest.mark.parametrize('weights',[[2,5,3],[0,5,3],None,'missing'])
def test_saved_valid_weights_match_constructor(weights):
    expected=ConfusionMatrix([0,1,1],[0,1,0],sample_weight=None if weights=='missing' else weights)
    actual=load(weights)
    assert actual.table==expected.table
    assert actual.weights==expected.weights
    assert actual.overall_stat==expected.overall_stat


@pytest.mark.parametrize('actual,predicted',[([0,1],[0,1,0]),([],[])])
def test_saved_invalid_vectors_rejected(actual,predicted):
    with pytest.raises(pycmVectorError): load(None,actual,predicted)
