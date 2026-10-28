import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from receipt_lab.common import strict_json, validate_schema
from receipt_lab.evaluate import score_one, summarize
from receipt_lab.prepare import target_from_ground_truth, UnsupportedAnnotation
from receipt_lab.serve import create_app


@pytest.fixture
def target():
    return {"items": [{"name": "Tea", "quantity": "1", "price": "10.00"}],
            "subtotal": None, "tax": None, "total": "10.00"}


def test_projection_uses_only_declared_fields(target):
    raw = {"gt_parse": {"menu": {"nm": "Tea", "cnt": "1", "price": "10.00", "sub_nm": "modifier"},
                        "total": {"total_price": "10.00", "cashprice": "20.00"}},
           "valid_line": [{"text": "must not become model input"}]}
    assert target_from_ground_truth(json.dumps(raw)) == target


def test_perfect_prediction(target):
    assert score_one(target, {"raw_text": json.dumps(target)})['field_f1'] == 1


def test_format_only_whitespace_allowed(target):
    prediction = {**target, "items": [{**target['items'][0], "name": "  Tea  "}]}
    assert score_one(target, {"raw_text": json.dumps(prediction)})['exact']


def test_wrong_amount_not_normalized_away(target):
    prediction = {**target, "total": "1000"}
    score = score_one(target, {"raw_text": json.dumps(prediction)})
    assert not score['total_correct']
    assert score['fp'] == 1 and score['fn'] == 1


def test_invalid_schema_and_truncation_are_failures(target):
    for prediction in [{"raw_text": "{}"}, {"raw_text": json.dumps(target), "truncated": True}, {"raw_text": "not json"}]:
        assert score_one(target, prediction)['field_f1'] == 0


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        strict_json('{"total":"1","total":"2"}')


def test_multivalue_annotations_not_silently_flattened():
    with pytest.raises(UnsupportedAnnotation):
        target_from_ground_truth(json.dumps({"gt_parse": {"total": {"total_price": ["55,834", "55,800"]}}}))


def test_line_order_is_part_of_metric(target):
    target = {**target, "items": target['items'] + [{"name": "Cake", "quantity": "2", "price": "20"}]}
    swapped = {**target, "items": list(reversed(target['items']))}
    score = score_one(target, {"raw_text": json.dumps(swapped)})
    assert not score['exact'] and score['field_f1'] < 1


def test_missing_predictions_count_against_metrics(target):
    rows = [{"id": str(i), "target": target, "width": 100, "height": 300} for i in range(2)]
    result = summarize(rows, {"0": {"raw_text": json.dumps(target), "latency_seconds": 2}})
    assert result['missing'] == 1
    assert result['document_exact_match'] == .5
    assert result['field_micro_f1'] == pytest.approx(2/3)


def test_local_api_valid_and_invalid_upload(target):
    class FakeExtractor:
        def predict(self, image):
            assert image.mode == "RGB"
            return {"parsed": target, "status": "ok"}
    client = TestClient(create_app(FakeExtractor()))
    image = io.BytesIO()
    Image.new('RGB', (10, 10)).save(image, format='PNG')
    assert client.post('/extract', files={'file': ('test.png', image.getvalue(), 'image/png')}).json()['parsed'] == target
    assert client.post('/extract', files={'file': ('bad.png', b'not an image', 'image/png')}).status_code == 400
