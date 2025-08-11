from src.llm_api_adapter.models.responses.chat_response import ChatResponse

def test_from_openai_response():
    api_response = {
        "model": "gpt-4",
        "id": "response123",
        "created": 1234567890,
        "usage": {"total_tokens": 100},
        "choices": [
            {
                "message": {"content": "Hello from OpenAI!"},
                "finish_reason": "stop"
            }
        ]
    }
    response = ChatResponse.from_openai_response(api_response)
    assert response.model == "gpt-4"
    assert response.response_id == "response123"
    assert response.timestamp == 1234567890
    assert response.tokens_used == 100
    assert response.content == "Hello from OpenAI!"
    assert response.finish_reason == "stop"

def test_from_anthropic_response(monkeypatch):
    api_response = {
        "usage": {"input_tokens": 30, "output_tokens": 70},
        "model": "claude-2",
        "id": "anthropic123",
        "content": [{"text": "Hello from Anthropic!"}],
        "stop_reason": "end"
    }
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    response = ChatResponse.from_anthropic_response(api_response)
    assert response.model == "claude-2"
    assert response.response_id == "anthropic123"
    assert response.tokens_used == 100
    assert response.content == "Hello from Anthropic!"
    assert response.finish_reason == "end"

def test_from_google_response():
    api_response = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello from Google!"}]},
                "finishReason": "length"
            }
        ],
        "usageMetadata": {"totalTokenCount": 42}
    }
    response = ChatResponse.from_google_response(api_response)
    assert response.tokens_used == 42
    assert response.content == "Hello from Google!"
    assert response.finish_reason == "length"
