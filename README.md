# Free-Model Aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/elpablo61/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/elpablo61/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Entry for the make.fun bounty **"The best live free-model aggregator"**
(`free-model-aggregator-02`, $500 prize pool, closes 17 Nov 2026 10:30 WIB).

One server, two APIs: point any OpenAI or Anthropic client at it and every
answer is a live completion from one of OpenRouter's fourteen free models,
dialled through the grader's gateway on the entrant's key. It routes each
prompt to the best-rated free model that can serve it, fails over to the
next free model when one refuses, and checks that the prompt's unique
token came back in the answer's text — retrying with an explicit
instruction when it did not.

## Run it

Node.js 22, standard library only, nothing to install:

```sh
node main.ts --list    # print the registered catalog document and exit 0
node main.ts           # serve (see below for the trial protocol)
```

In a trial the env provides `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`
(the brief: the free-model list and the six prompts),
`OPENAI_BASE_URL` (the gateway root) and `OPENAI_API_KEY` (the trial's
dummy key, swapped by the gateway for the entrant's OpenRouter key). The
server listens on an ephemeral port, registers with the env
(`POST /register {"protocol_version": 1, "port", "catalog"}`), serves
`GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/messages`
(JSON or server-sent events), polls `GET /work` every 100 ms and exits 0
when the env says `{"done": true}`.

Without the trial env vars it serves standalone on `$PORT` (default
8080), reading the gateway from `OPENAI_BASE_URL` (else
`OPENROUTER_BASE_URL`, else `https://openrouter.ai/api/v1`) and the key
from `OPENAI_API_KEY` (else `OPENROUTER_API_KEY`).

## Routing — rating-first, capability-gated

The brief lists the fourteen free models with the house's quality
ratings (`rating_milli`). The server ranks the models that meet a
prompt's requirements by rating, best first:

| requirement | first pick | rating | next |
|---|---|---|---|
| JSON mode | `openrouter/nvidia/nemotron-3.5-super-vl:free` | 1303 | gemma-4-31b (1210), then the 1150s |
| anything else (plain, tools, streaming) | `openrouter/thinkingmachines/inkling-2-small:free` | 1475 | super-vl (1303), ultra (1285), lightning (1221) |

A prompt's requirements come from the brief's prompt rows, in arrival
order (the body's own requirements are the fallback when the brief has
no row for a request). A model that refused on the previous prompt is
skipped for the next one — a count-based cooldown, no clock, so a replay
leaves the same state export.

## Failover and the token

- A gateway 400/404/429/5xx fails over to the next eligible model;
  401/403 fail fast and are passed back as the provider sent them, with
  no `retry-after` header.
- Refusals are reported on the response in the
  `x-makefun-upstream-failures` header (a JSON array of at most 8
  `{"status", "code", "model"}` rows, in refusal order), as the brief
  allows — it is the only way the failover column can see a refusal the
  server retried and then served.
- Each prompt carries one unique token (`WORD-1234`). The server checks
  the token in the answer's **text** (what the house checks — never in
  reasoning or tool arguments, and never written in by the server: the
  house scores the provider's own answer body). If the token is missing,
  it retries (at most twice) with an explicit instruction naming the
  missing token verbatim, so the model itself produces it. At most 6
  gateway calls per prompt; a prompt's wall time is capped near the
  driver's 30 s timeout.
- A request naming a model outside the free set is refused
  `400 model_not_allowed`; `model: "auto"` (what the trial sends) means
  the server picks.

## Wire fidelity

- `GET /v1/models` answers in the shape of its auth header:
  `Authorization: Bearer` → the OpenAI list shape,
  `x-api-key` → the Anthropic list shape. Both list exactly the fourteen
  free ids, once each — the same set the catalog registers.
- OpenAI streaming is real SSE: role chunk, content chunks, an optional
  tool-call chunk, one finish chunk, a usage chunk (`choices: []`), then
  `data: [DONE]`. Anthropic streaming is the `message_start` /
  `content_block_start` / `content_block_delta` / `content_block_stop` /
  `message_delta` / `message_stop` sequence, with usage and the stop
  reason translated (`tool_calls` → `tool_use`, `length` →
  `max_tokens`, `content_filter` → `refusal`).
- The `model` on every answer and every streamed chunk is the id the
  server actually dialled (`openrouter/…:free`), never the provider's
  bare spelling.
- Deterministic: ids come from a counter, `created` is fixed, usage
  comes from the gateway (or a character-count estimate). No wall clock
  and no unseeded randomness in any response.

## Local verification

Tested against the kit's real env server and scorer with a mock
gateway (see `BUILD_LOG.md`, kept out of the repo): all three practice
trials 6/6 valid, 39/39 checks, optimal routing; a 429 injected on the
best model fails over and is reported in the trail header; an answer
missing its token is retried with the instruction and served; the state
export is byte-identical across runs. Live-model quality (60% of the
score) cannot be measured locally — no OpenRouter key exists in this
environment; it needs the key pushed to the competition before submit.

## Submission plan

- GitHub repository for the entry: elpablo61/free-model-aggregator-02,
  created after the bounty opened (15 Nov 2026 10:30 UTC), MIT license
  included.
- make.fun account: @pablo-crypto (the wallet-only account
  sage-marten-59 was merged into it 16 Nov 2026; the entry's owner
  wallet is `H7gLhKDVvKkf582ypnbDVKQ7CTx3VWR61z7mVJ4cZ9hm`, see
  `.makefun/owner.json`).
- No submission has been made from this build yet.

---

<sub>Entered on [make.fun](https://make.fun/?ref=f5mdkdx2&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
