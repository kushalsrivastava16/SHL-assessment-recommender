from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

Role = Literal["user", "assistant", "system"]

# SHL catalog test-type keys.
TEST_TYPE_KEYS = {"A", "B", "C", "D", "E", "K", "P", "S"}


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def non_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages must contain at least one turn")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # one or more keys, e.g. "K" or "P" (or "C, P")


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def cap(cls, v: List[Recommendation]) -> List[Recommendation]:
        # Never emit more than 10; the spec forbids it.
        return v[:10]


class HealthResponse(BaseModel):
    status: str = "ok"
