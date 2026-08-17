from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://fireworksmf.services.ai.azure.com/openai/v1"
deployment_name = "Kimi-K2.6"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)

if completion.usage is not None:
    print(f"Input tokens: {completion.usage.prompt_tokens}")
    print(f"Output tokens: {completion.usage.completion_tokens}")
    print(f"Total tokens: {completion.usage.total_tokens}")

details = completion.usage.completion_tokens_details

if details is not None:
    print(f"Reasoning tokens: {details.reasoning_tokens}")