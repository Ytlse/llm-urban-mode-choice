# Add a provider

A *provider* is a named instance in the providers file designated by `LLM_GATEWAY_PROVIDERS_FILE` (in this repository: `config/llm_gateway/providers.yaml`; the package only ships a `providers.example.yaml`): a
model, a URL, quotas, a weight. Several instances can share the same
*adapter* (the translator to the provider's API). Three cases, from the simplest to the rarest.

## Case 1 — one more instance on an existing adapter

Shipped adapters: `openai`, `mistral`, `google`, `groq`, `cerebras`. Add a block under
`providers:`:

```yaml
  groq_qwen_qwen3_6_27b_key1:
    adapter:                groq                       # nom de l'adapter (défaut = nom de l'instance)
    rpm_limit:              30
    tpm_limit:              8000
    rpd_limit:              1000
    max_tokens_per_request: 8000                       # requête unique > TPM → HTTP 413 chez Groq
    base_url:               https://api.groq.com/openai/v1
    default_model:          qwen/qwen3.6-27b
    weight:                 0.18                       # min(30, 8000/3000)/15
    concurrency_limit:      3
    disable_timeout:        120
```

Then the key: `PROVIDER_KEYS__groq_qwen_qwen3_6_27b=…` or, more often, the adapter's shared
key `PROVIDER_KEYS__groq=…`. The resolution is `provider_keys[nom_instance] or
provider_keys[adapter]`: an instance whose name matches no variable falls back
on the adapter's key, silently. An instance with no key at all is excluded at startup
("Provider 'x' excluded: missing API key.").

Check:

```bash
llm-gateway config validate        # OK — N provider(s) avec clé : …
llm-gateway config show            # batch_max_agents et tpm_estimate_per_request calculés, secrets masqués
make providers DRY_RUN=1           # from the root: probes the real quotas without writing
```

## The fields, and who applies them

| Field | Required | Default | Applied by |
|---|---|---|---|
| `rpm_limit` | yes | — | rate-limiter: 60 s sliding window, smoothing `min_interval = 60 / rpm` (a reservation too close to the previous one is refused) |
| `base_url` | yes | — | adapter (`_get_base_url`) |
| `wait_timeout` | no | client default | **no one on the gateway side**: copied as is and published, the SDK receives it per call (`execute(wait_timeout=…)`). For a local model, a task's wait is that of the queue, not of the generation. |
| `default_model` | yes | — | adapter (`_resolve_model`) if the request does not impose one |
| `adapter` | no | instance name | `get_adapter`: adapter class and key inheritance `PROVIDER_KEYS__<adapter>` |
| `tpm_limit` | no | `None` | rate-limiter: reservation of estimated tokens in the same 60 s window; also bounds `batch_max_agents` |
| `rpd_limit` | no | `None` | rate-limiter: requests/day counter in the provider's time zone; reached → provider set aside until its reset |
| `quota_reset_tz` | no | `UTC` | time zone where the provider places midnight (`America/Los_Angeles` for Google) |
| `tpd_limit` | no | `None` | rate-limiter: tokens/UTC day counted afterwards (actual tokens); same setting aside |
| `max_tokens_per_request` | no | `None` | startup: exclusion if `< batch_max_agents × assumed_prompt_tokens + min_output_tokens`; worker: 413 safeguard on the rendered prompt (`ProviderCapacityError`) |
| `max_output_tokens` | no | `None` | worker: bounds the `max_tokens` sent; balancer: excludes the provider if `< parameters.max_tokens` of the request; **learned** on HTTP 400 and rewritten in the file |
| `weight` | no | `1.0` | balancer: SWRR sequence; `0` = defined but **out of rotation**, usable only with `force_provider` |
| `concurrency_limit` | no | `2` | rate-limiter: simultaneous Celery workers on this provider |
| `disable_timeout` | no | `180` | worker: deactivation duration after 30 consecutive errors |
| `batch_max_agents` | do not write | computed | `compute_batch_max_agents` (core/batching), published in `/health` — see [Tune the quotas](regler-les-quotas.md) |
| `tpm_estimate_per_request` | do not write | computed | same |

All these values are **applied** by the code. The quotas themselves, however, are
readings: a `# TBC` in the file flags an unverified value, and the dated
comments (`vérifié 2026-08-03 (en-têtes x-ratelimit)`) say where the
figure comes from. `make providers` (`scripts/providers/refresh.py`) refreshes them from the
actual headers.

!!! warning "A misspelled field is ignored without error"
    `ProviderConfig` currently accepts unknown keys (default pydantic behaviour,
    `extra` not set): `rmp_limit: 30` raises nothing, and the missing required `rpm_limit`
    field will raise a `ValidationError` only if no other line
    provides it. Reread `llm-gateway config show` after an edit.

## Weight convention

The weight is proportional to the capacity actually sustainable, not to the advertised RPM:

```
weight = min(rpm_limit, tpm_limit / 3000) / 15
```

3,000 ≈ tokens (input + output) of an average request, 15 = reference RPM (weight 1.0).
For a provider with a small TPM it is the TPM that bounds: `groq_openai_120_key1` advertises 30 RPM but
8,000 TPM only sustains ~2.7 requests/min, hence `0.18`. Recompute at each change of
`rpm_limit` or `tpm_limit`; `make providers` does it. The readjustment of 2026-07-10 showed
what is at stake: Mistral carried 47 % of the total capacity and received 8 % of the traffic.

The SWRR sequence ([explanation](../explications/batching-swrr-disjoncteur.md)) distributes
100 slots in proportion to the weights, each provider in rotation having at least one.

## Case 2 — a second key for the same provider

Google free-tier quotas are counted **per project and per model**. A key from another
project is another bucket of 500 requests/day: that is the `google_gemini31_key2` instance (key
`PROVIDER_KEYS__google2`), and `google_gemini35_key2` reuses the same key on the 3.5 model
(`docker-compose.yml` derives `PROVIDER_KEYS__google2_35` from `PROVIDER_KEYS__google2` rather
than duplicating the secret in `.env`).

Name the instance `<modèle>_key<N>`: the suffix makes the physical key readable, and it **forbids
the fallback** to `PROVIDER_KEYS__<adapter>`. Without `PROVIDER_KEYS__<nom_instance>`, the instance is
set aside from the rotation and startup reports it — instead of silently serving key 1, which
has already sent calls to the wrong key. So add the mapping line in
`docker-compose.yml` at the same time as the instance in `providers.yaml`.

For a provider that does not place midnight in UTC, also set
`quota_reset_tz` (see `regler-les-quotas.md`).

## Case 3 — an OpenAI-compatible API, without writing code

Ollama, vLLM, OpenRouter, LM Studio, Together, Fireworks… speak the
`POST {base_url}/chat/completions` dialect: that is the `openai_compatible` translator, of which OpenAI, Groq,
Cerebras and Mistral are only settings. An entry in the providers file is enough:

```yaml
mon_ollama:
  adapter: openai_compatible
  structured_output: json_object     # json_schema (natif OpenAI) | json_object | none
  schema_in_system: true             # copies the JSON schema into the system message
  base_url: http://ollama:11434/v1
  default_model: qwen3:8b
  rpm_limit: 60
  weight: 1.0
```

then the key `PROVIDER_KEYS__mon_ollama` (any value if the API does not require one: an
instance without a key is excluded from the rotation). The two adapter settings also apply to the
four shipped dialects: `mistral` with `structured_output: json_schema` switches a recent model
to native structured output, without code.

!!! note "LM Studio does not accept `json_object`"
    Checked on 2026-09-08: `response_format.type` must be `json_schema` or `text`, otherwise
    HTTP 400 "'response_format.type' must be 'json_schema' or 'text'". So declare
    `structured_output: json_schema`. From a container, LM Studio on the host is reached via
    `http://host.docker.internal:1234/v1`, and `default_model` is the identifier from `lms ls`.
    Set `weight: 0` if only one model is loaded at a time: out of rotation, the instance only serves
    the requests that force it (`force_provider`), and the cascade does not trigger
    on-the-fly loading. Reference deployment: LM Studio section of
    `config/llm_gateway/providers.yaml` and `docs/setup/llm-providers.md`.

| Shipped dialect | `structured_output` | `schema_in_system` |
|---|---|---|
| `openai` | `json_schema` | no |
| `groq` | `json_object` | no (the schema comes from the prompt) |
| `cerebras` | `json_object` | yes |
| `mistral` | `json_object` | yes |

Network errors (timeout, connection refused, truncated response) are classified by the base as
`ProviderServerError` with `error_type` `network_timeout`, `network_connect`, `network_protocol`:
the worker retries them with backoff and cooldown, like a 5xx.

## Case 4 — a new adapter, in your package

When the API is compatible with no known dialect. Subclass `BaseAdapter` (or
`OpenAICompatibleAdapter` if only a detail differs: override `build_payload` or `_headers`):

```python
from llm_gateway.adapters.base import BaseAdapter
from llm_gateway.core.models import InternalRequest, LLMOutput


class MonAdapter(BaseAdapter):
    provider_name = "monfournisseur"     # = value of the `adapter` field in the providers file
    request_timeout = 120.0              # override if the provider is slow (Google: 240 s)

    def call(self, request: InternalRequest) -> tuple[LLMOutput, int, int]:
        response = self._post(                              # erreurs réseau → ProviderServerError
            f"{self._get_base_url()}/…",
            headers={"Authorization": f"Bearer {self._get_api_key().get_secret_value()}"},
            json={"model": self._resolve_model(request), "prompt": "…", "top_p": request.top_p},
        )
        self._raise_for_status(response)                    # 5xx → ProviderServerError, 4xx → ProviderClientError
        data = response.json()
        return self._parse_output(data["text"]), data["in"], data["out"]
```

The contract of `call`: return `(LLMOutput, tokens_in, tokens_out)` or raise
`ProviderServerError` (replayed with backoff), `ProviderClientError` (4xx), `ProviderParseError`
(JSON outside the schema). `_parse_output` does the tolerant cleaning (Markdown fences, `json-repair`
repair, repetition loop, `agents` key); `_raise_for_status` captures the retry
delay of 429s; `request.top_p` can be `None`: only send it if it is defined.

Declare the class in the `pyproject.toml` of **your** package, the gateway discovers it at the
first call without anything changing on its side:

```toml
[project.entry-points."llm_gateway.adapters"]
monfournisseur = "mon_paquet.adapters:MonAdapter"
```

The entry point name becomes the `provider_name` if the class does not set one. An
unloadable adapter is logged as `[ALARME]` and does not prevent the others from serving. Test on a
doubled httpx client, in the manner of `tests/integration/test_openai_compatible_adapter.py`, and
add an entry to the corpus `tests/data/llm_outputs.json` if the provider produces outputs
of a new shape.

## Remove a provider

Comment out the block rather than deleting it, with the date and the reason (`# [obsolète
2026-08-18] default_model … absent de /models`). `make providers` never re-adds a
model already present in the file, even commented out: a commented block is a decision.
