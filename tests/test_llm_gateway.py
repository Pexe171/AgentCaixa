import httpx

from rag_app.agent.llm_gateway import OpenAILLMGateway, _request_with_retries


def test_openai_gateway_supports_tool_calling_roundtrip() -> None:
    gateway = OpenAILLMGateway(api_key="token", model="gpt-test")

    responses = [
        {
            "id": "resp_1",
            "output_text": "",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "consultar_caixa",
                    "arguments": '{"id_cliente":"123"}',
                }
            ],
        },
        {
            "id": "resp_2",
            "output_text": "Cliente 123 está adimplente.",
            "output": [],
        },
    ]
    sent_payloads: list[dict[str, object]] = []

    def fake_send_request(payload: dict[str, object]) -> dict[str, object]:
        sent_payloads.append(payload)
        return responses.pop(0)

    gateway._send_request = fake_send_request  # type: ignore[method-assign]

    captured_call: dict[str, object] = {}

    def tool_executor(name: str, arguments: dict[str, object]) -> dict[str, str]:
        captured_call["name"] = name
        captured_call["arguments"] = arguments
        return {"status": "ok"}

    output = gateway.generate(
        system_prompt="sistema",
        user_prompt="usuario",
        tools=[{"type": "function", "name": "consultar_caixa"}],
        tool_executor=tool_executor,
    )

    assert output.text == "Cliente 123 está adimplente."
    assert captured_call == {
        "name": "consultar_caixa",
        "arguments": {"id_cliente": "123"},
    }
    assert sent_payloads[1]["previous_response_id"] == "resp_1"


def test_request_with_retries_retries_on_429_and_succeeds() -> None:
    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.request = httpx.Request("POST", "https://example.com")

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "error",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, object],
        ) -> FakeResponse:
            del url, headers, json
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(status_code=429, payload={})
            return FakeResponse(status_code=200, payload={"ok": True})

    client = FakeClient()

    payload = _request_with_retries(
        client=client,
        url="https://example.com",
        headers={"Content-Type": "application/json"},
        payload={"ping": "pong"},
        max_attempts=3,
        initial_backoff_s=0.001,
    )

    assert payload == {"ok": True}
    assert client.calls == 2
