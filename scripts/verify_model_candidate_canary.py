"""Two-model transport canary. Returns tool requests but executes no tools."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-source", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--gateway-config", required=True)
    args = parser.parse_args()
    source = Path(args.transport_source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != args.expected_source_sha256:
        raise SystemExit("canary_source_integrity_failed")
    import yaml
    config = yaml.safe_load(Path(args.gateway_config).read_text(encoding="utf-8-sig"))
    keys = config.get("api-keys") or []
    if len(keys) != 1 or not isinstance(keys[0], str) or not keys[0]:
        raise SystemExit("local_gateway_client_key_unavailable")
    spec = importlib.util.spec_from_file_location("evomind_runtime.model_transport_canary", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from research_os.agent.messaging import AgentMessageClient, OpenAITransport, ToolSpec
    from research_os.llm_client import ProviderConfig

    def check(model):
        events = []
        base = AgentMessageClient(max_retries=2, timeout=45, transports=[OpenAITransport(
            ProviderConfig("openai", "http://127.0.0.1:65068/v1", model, keys[0], reasoning_effort="low", service_tier="priority"))])
        client = module.governed_client(base, events.append)
        started = time.monotonic()
        result = {"requested_model": model, "scope": "native_tool_transport_canary_only", "tools_executed": 0,
                  "duration_90_minutes_verified": False, "qualifies_as_default": False}
        try:
            turn = client.send([{"role": "user", "content": "Call contract_probe once with value 37. Do not produce prose."}],
                system="This is an isolated API transport check. The requested tool is a fixture and will not be executed.",
                tools=[ToolSpec("contract_probe", "Return a fixed probe value", {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False})])
            result.update(status="passed" if len(turn.tool_calls) == 1 and turn.tool_calls[0].name == "contract_probe" and turn.tool_calls[0].input == {"value": 37} else "failed",
                          reported_model=turn.model, native_calls_returned=len(turn.tool_calls),
                          input_tokens=turn.input_tokens, output_tokens=turn.output_tokens)
        except Exception as error:
            result.update(status="failed", error_class=type(error).__name__, error_code=getattr(error, "code", "canary_error"))
        result.update(elapsed_seconds=time.monotonic()-started, attempts=events,
                      transport_source_sha256=args.expected_source_sha256)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, ("gpt-6-astra", "gpt-5.6-sol")))
    print(json.dumps({"schema": "evomind.two_model_transport_canary.v1", "checks": results,
                      "production_default_changed": False, "gpu_actions": 0, "credentials_emitted": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
