import pytest
import numpy as np
import json
from inference.inference import input_fn, output_fn

def test_input_fn_csv():
    body = "0,12,1,0,1,1,2,1,1,1,1,1,0,1,1,3,1,65.5,850.5"
    result = input_fn(body, "text/csv")
    assert result.shape == (1, 19)

def test_input_fn_multiple_rows():
    body = "0,12,1,0,1,1,2,1,1,1,1,1,0,1,1,3,1,65.5,850.5\n1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,29.85,29.85"
    result = input_fn(body, "text/csv")
    assert result.shape == (2, 19)

def test_input_fn_wrong_content_type():
    with pytest.raises(ValueError):
        input_fn("data", "application/xml")

def test_output_fn_csv():
    prediction = np.array([[0, 0.1842]])
    result, content_type = output_fn(prediction, "text/csv")
    assert content_type == "text/csv"
    assert "0,0.1842" in result

def test_output_fn_json():
    prediction = np.array([[1, 0.75]])
    result, content_type = output_fn(prediction, "application/json")
    assert content_type == "application/json"
    data = json.loads(result)
    assert data[0]["prediction"] == 1
    assert data[0]["label"] == "Churn"
    assert data[0]["churn_probability"] == 0.75

def test_output_fn_no_churn():
    prediction = np.array([[0, 0.18]])
    result, content_type = output_fn(prediction, "application/json")
    data = json.loads(result)
    assert data[0]["label"] == "No Churn"

def test_output_fn_wrong_accept():
    with pytest.raises(ValueError):
        output_fn(np.array([[0, 0.5]]), "text/xml")
