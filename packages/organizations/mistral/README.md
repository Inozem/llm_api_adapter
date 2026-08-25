# llm-api-adapter-mistral

Official direct-API support for Mistral in
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

Supported models:

- `mistral-small-2603`
- `mistral-medium-3-5`
- `mistral-large-2512`

## PDF input

`DocumentPart` supports PDF URLs and bytes. Before the chat request, the
adapter sends each PDF to Mistral OCR 4.1 (`mistral-ocr-4-1`) and supplies the resulting Markdown to
the selected chat model. This creates a separate OCR API request, subject to
Mistral's OCR limits and pricing; `ChatResponse` usage and cost cover only the
chat completion.

Install the package alongside a compatible core distribution:

```bash
pip install llm-api-adapter-mistral
```
