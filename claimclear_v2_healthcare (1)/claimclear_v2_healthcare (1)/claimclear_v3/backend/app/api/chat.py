"""Chat route — multi-turn RAG conversation about a claim"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_service import chat_with_claim

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        reply, suggested, sources = await chat_with_claim(
            claim_context=body.claim_context,
            messages=body.messages,
            policy_id=body.policy_id,
        )
        return ChatResponse(
            reply=reply,
            suggested_questions=suggested,
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
