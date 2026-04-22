import google.auth, google.auth.transport.requests
from openai import OpenAI

creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
creds.refresh(google.auth.transport.requests.Request())

client = OpenAI(
    base_url="https://us-central1-aiplatform.googleapis.com/v1/projects/finindiabench/locations/us-central1/endpoints/openapi",
    api_key=creds.token,
)
r = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[{"role": "user", "content": "Reply with just the word: OK"}],
    max_tokens=10,
)
print(r.choices[0].message.content)