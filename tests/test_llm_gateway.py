from rag_app.agent.llm_gateway import OpenAILLMGateway


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
