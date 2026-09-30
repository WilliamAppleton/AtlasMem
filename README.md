# AtlasMem

Just-in-Time Memory for LLM agents, after
[*Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents*](https://arxiv.org/abs/2609.27334)
(Zhou et al., 2026).

Most agent memory systems decide what to keep when a task **ends**. AtlasMem keeps the
raw trajectory and defers curation to **read time**, when the next task is known:

```
write:  finished trajectory ──► judge (success?) ──► store raw in SQLite
read:   new task ──► BM25 over stored task descriptions ──► top-k raw trajectories
                 ──► curator LLM ──► short task-specific briefing ──► prepended to executor prompt
```

The briefing is ephemeral; only raw trajectories are stored. This version uses an
**untrained, prompted curator**, which the paper reports is already competitive with
write-time memory systems. Training the curator with GRPO is future work.

No runtime dependencies (stdlib only). LLMs are called through Ollama (local or
Ollama Cloud) or any OpenAI-compatible server (e.g. vLLM).

## Install

```bash
uv venv && uv pip install -e '.[dev]'      # add ,mcp for the MCP server
```

## Configuration

| Env var | Default |
|---|---|
| `OLLAMA_HOST` | `https://ollama.com` (use `http://localhost:11434` for local Ollama) |
| `OLLAMA_API_KEY` | unset; sent as bearer token when set |
| `ATLASMEM_DB` | `~/.atlasmem/memory.db` |
| `ATLASMEM_CURATOR_MODEL` | `gpt-oss:20b` |
| `ATLASMEM_JUDGE_MODEL` | `gpt-oss:20b` |
| `ATLASMEM_EXECUTOR_MODEL` | `gpt-oss:120b` (demo only) |
| `ATLASMEM_K` | `3` |

## Library

```python
from atlasmem import build_memory, Trajectory, Step

mem = build_memory()                       # SQLite + BM25 + Ollama curator/judge
b = mem.brief("put a clean mug in shelf 1")
system_prompt = (b.payload + "\n\n" + base_prompt) if b.payload else base_prompt

# ... agent runs ...
traj = Trajectory(task="put a clean mug in shelf 1",
                  steps=[Step(observation="...", action="...")])
mem.record(traj)               # LLM judge gates the write
mem.record(traj, label=True)   # or use a ground-truth label
```

Pieces are swappable: `MemoryBank`, `BM25Retriever`, `Curator`, `LLMJudge`, and any
object with `chat(messages) -> str` works as an LLM (`OllamaLLM`, `OpenAICompatLLM`).
`build_curator_messages()` is the exact curator prompt, exposed for later GRPO training.

## CLI

```bash
atlasmem brief "put a hot apple in cabinet 1"     # retrieve + curate
atlasmem record traj.json [--label success]       # JSON object or list; '-' for stdin
atlasmem list | show <id> | delete <id> | delete --all
atlasmem serve                                    # MCP server over stdio (needs [mcp])
```

## Demo: kitchen environment

`atlasmem.envs.KitchenEnv` is a small ALFWorld-style text world whose object layout is
fixed across tasks, so memory of earlier episodes really helps later ones.

```bash
atlasmem demo --episodes 20 --no-memory             # baseline
atlasmem demo --episodes 20 -v                      # with JIT memory (LLM judge)
atlasmem demo --episodes 20 --ground-truth --batch-size 4
```

`--batch-size` follows the paper's protocol: tasks in a batch share one bank state and
the bank is updated after each batch. The demo bank is in-memory unless `--db` is given.

## MCP server

Tools: `get_briefing(task, k)`, `record_trajectory(task, steps, success?)`, `memory_stats()`.

```json
{"mcpServers": {"atlasmem": {"command": "atlasmem", "args": ["serve"]}}}
```

## Tests

```bash
pytest                      # offline, uses fake LLMs
ATLASMEM_LIVE=1 pytest -m live   # hits Ollama
```
