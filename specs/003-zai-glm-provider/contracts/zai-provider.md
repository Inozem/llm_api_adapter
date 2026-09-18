# Public Contract: Z.ai / GLM Provider

## Installation and selection

```bash
pip install "llm-api-adapter[zai]"
# or: pip install llm-api-adapter-zai
```

```python
UniversalLLMAPIAdapter(
    organization="zai",
    model="glm-5.3-flash",
    api_key=os.environ["ZAI_API_KEY"],
)
```

The public facade remains unchanged. Selecting `zai` without its package returns the established
installation remedy. Arbitrary endpoints and deployment settings are not public inputs.

## Compatibility matrix

| Capability | Contract |
| --- | --- |
| Completed chat, async chat, sync/async stream | Supported through normalized existing transports |
| Tools | Auto choice only, up to 128 declarations |
| Reasoning | `low`, `high`, `max`; never emitted as visible text |
| Image parts | URL and byte/data-URL forms after deterministic and live proof |
| Portable structured output | Rejected before outbound HTTP |
| Documents | Rejected before outbound HTTP until both direct forms pass release E2E |
| Unknown model/capability | No inference; documented validation error |

## Release boundary

The package calls only the official PaaS endpoint. It adds no SDK, upload, OCR, URL fetch, retry,
persistence, continuation metadata, video, or deployment support. API keys and raw reasoning/tool
data never appear in fixtures, docs, logs, or diagnostics.

The provider may be announced only after every applicable shared Core E2E scenario and
package-local boundary check pass against exact candidate artifacts. Native provider success does
not substitute for the declared portable contract.
