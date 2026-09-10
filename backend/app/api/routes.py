from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import Evidence, HealthResponse, PredictRequest, PredictResponse
from app.services.escalation import EscalationService
from app.services.intent_classifier import IntentClassifier
from app.services.response_generator import ResponseGenerator
from app.services.retrieval import RetrievalService

router = APIRouter()
settings = get_settings()
classifier = IntentClassifier()
retrieval = RetrievalService(settings)
generator = ResponseGenerator(settings)
escalation = EscalationService(settings)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", brand=settings.selected_brand or None)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    prediction = classifier.predict(request.message)
    examples = retrieval.retrieve_similar_examples(request.message)
    reply = generator.generate(request.message, examples)
    decision, reason = escalation.decide(request.message, prediction, examples, reply)
    if reply is None:
        reply = "I could not find enough historical guidance to answer this reliably. A support specialist should review this request."
    return PredictResponse(
        intent=prediction.intent,
        intent_confidence=prediction.confidence,
        reply=reply,
        decision=decision,
        reason=reason,
        evidence=[Evidence(**example) for example in examples],
    )
