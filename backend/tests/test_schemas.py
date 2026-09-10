import pytest
from pydantic import ValidationError

from app.models.schemas import PredictRequest, PredictResponse


def test_predict_request_requires_message():
    with pytest.raises(ValidationError):
        PredictRequest(message="")


def test_predict_response_contract():
    response = PredictResponse(intent="baggage", intent_confidence=0.8, reply="We can help.", decision="AUTO_HANDLE", reason="Evidence found")
    assert response.evidence == []
