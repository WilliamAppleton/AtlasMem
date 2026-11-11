# Hybrid memory: dreaming between batches, just-in-time curation at read time (design)

Status: draft, 11 November 2026. Nothing here is implemented yet.

**Summary:** keep AtlasMem's raw store, evidence-checked write gate and read-time curator, and add
a *dream* step between batches. The dream step turns many trajectories into a small set of standing
notes. Every note must quote observations, and the quotes are checked in code just as the judge's
are. At read time the curator writes the briefing from those notes plus the top-k trajectories.
Then run a paired experiment to see whether the hybrid beats plain JIT memory on the kitchen world,
including a run where the layout changes partway through.

## Why

The two approaches curate at different times, so they can be combined rather than chosen between.

- **JIT memory** (Zhou et al., 2026, arXiv:2609.27334), as AtlasMem implements it, curates at read
  time, once the next task is known. The curator only sees the k trajectories that BM25 finds by
  *task description*.
- **Anthropic's "The components of agent memory"** (6 November 2026) has the model decide what is
  worth saving, then runs an offline "dreaming" pass over transcripts. That pass fixes wrong notes,
  builds larger representations such as wikis, and promotes key notes into a small layer that is
  always loaded.

Each approach covers a weakness of the other:

1. **The curator misses incidental knowledge.** A trajectory for "put a hot apple in shelf 1" may
   open cabinet 3 and see the mug there. That trajectory will never be retrieved for "put a clean
   mug in drawer 1", because BM25 only indexes task text. So the curator reports the mug's location
   as unknown even though the bank holds it. Dreaming reads every trajectory, so it can pull that
   observation out.
2. **Dreaming can lock in mistakes.** It can turn an error into standing policy, and a wrong note in
   the always-loaded layer misleads every later task. AtlasMem's main asset is that it won't trust
   a claim it can't trace to an observation. Applying the same rule to notes makes dreaming safer.
3. **Curation is repeated on every read.** The curator works out the same stable facts (the layout,
   which command makes an object hot) task after task. Dreaming works them out once per batch.

## Current behaviour that this changes

From `src/atlasmem` at commit `9aafcfa`:

- `JitMemory.record` / `record_batch` store a trajectory **only if** the verdict is SUCCESS, so
  failures are discarded. Only `judge_reason` survives into `metadata`. Evidence quotes and the
  `overruled` flag are lost.
- `BM25Retriever` indexes `bank.tasks()`, which is task descriptions only.
- `Curator` sees only the retrieved trajectories.
- `run_sequence` already uses the paper's batch protocol: every task in a batch sees one bank
  state, and the bank updates after the batch. That gap between batches is where the dream step
  goes.

## Design

### 1. Store every trajectory, along with its verdict

Dreaming needs failures. They show traps, and their observations are still true facts about the
world even when the task failed.

- `MemoryBank` gains a `verdict INTEGER` column (1 = judged success, 0 = judged failure, NULL =
  never judged). Add it with `ALTER TABLE` if it's missing, so existing databases keep working.
  The existing `success` column keeps meaning "environment signal".
- `record` / `record_batch` store **all** trajectories. The full `Verdict` (success, reason,
  evidence, overruled) goes into `metadata["verdict"]`.
- `BM25Retriever` gains `verified_only: bool = True`. That keeps today's read behaviour: the
  curator still sees only verified successes.
- Add `MemoryBank.since(version_or_time)`, so a dream can read just the trajectories added since
  the last dream.

`RecordResult.stored` keeps its meaning ("eligible for retrieval"), so callers don't break.

### 2. Notes

A new `notes` table in the same SQLite file:

| Column | Meaning |
|---|---|
| `id` | uuid hex |
| `kind` | `fact`, `procedure` or `trap` |
| `text` | one sentence, e.g. "The mug starts in cabinet 3." |
| `evidence` | JSON list of `{traj_id, step, quote}` |
| `status` | `active` or `retired` |
| `retired_reason` | free text, for audit |
| `created_at`, `updated_at` | timestamps |
| `support` | number of distinct trajectories whose evidence backs the note |

What each kind of note may cite, enforced in code:

| Kind | May cite | Rationale |
|---|---|---|
| `fact` | observations from any trajectory | The environment said it. The task outcome doesn't change that. |
| `procedure` | observations from **verified-success** trajectories only | A strategy is only proven if it worked. |
| `trap` | observations from **failure** trajectories | E.g. "a mug is not a cup", "cabinet 2 must be opened first". |

Notes never cite actions, for the same reason the judge doesn't.

### 3. The dream step

`atlasmem.dream.Dreamer(llm, max_notes=25, max_words=300)` exposes
`dream(bank, notes, since) -> DreamReport`.

1. **Input:** the active notes, plus every trajectory added since the last dream (successes and
   failures, each tagged with its verdict). If a batch is too large, chunk it.
2. **The LLM proposes operations,** in a strict line format like the judge's:
   `ADD <kind> | <text> | EVIDENCE: <traj_id>#<step> "<quote>"`,
   `UPDATE <note_id> | <new text> | EVIDENCE: ...`,
   `RETIRE <note_id> | REASON: ... | EVIDENCE: ...`.
   A RETIRE must quote the newer observation that contradicts the note.
3. **Code checks each operation.** The quote must appear (via `judge._norm`) in that step's
   *observation* of the cited trajectory. The trajectory's verdict must also fit the note's kind
   (table above). Operations that fail are dropped and counted in the report. Nothing unverified
   reaches the notes table.
4. **Budget.** If the active notes exceed `max_notes` or `max_words`, a second call asks for
   MERGE/RETIRE operations only, under the same evidence rules. This is Anthropic's main lesson
   about note volume: in their Pokémon runs, a 413-file memory stalled the agent, while a run with
   about ten files did far better.
5. **Report:** proposed, accepted and rejected operations by reason, plus tokens and latency. The
   report is printed and saved with eval results.

The dream prompt says what notes are *for* ("facts and procedures that will help future, unknown
tasks in this environment"). It does not prescribe what to record. Anthropic found that a fixed
"record only X" rule reduced the gain for stronger models.

### 4. Read path

`JitMemory.brief` gains `notes: NoteStore | None`.

- **Hybrid:** the curator prompt gets a `STANDING NOTES` section ahead of the retrieved memories.
  The curator rules gain one line: notes are verified and may be relied on, and when a note
  conflicts with a retrieved memory, the more recent evidence wins. The briefing budget stays at
  120 words, so the curator still selects and adapts rather than copying everything.
- **Dream-only arm (experiment only):** active notes go straight into the executor's payload with
  no curator call. This approximates Anthropic's always-loaded layer.
- `build_curator_messages` stays a pure function (it gains a `notes` argument), so the GRPO plan is
  unaffected.

### 5. Feedback logging

Each episode's trajectory `metadata` records `briefing` (the payload text), `note_ids` (notes
shown to the curator) and `retrieved_ids`. This costs nothing now. Later it lets a dream retire
notes that keep preceding failures, and it is the (briefing, outcome) data that curator training
will need.

### Constraints

- Stdlib only, as now. Notes live in the same SQLite file.
- The dream model is set like the others: `ATLASMEM_DREAM_MODEL`, defaulting to `gpt-oss:120b`
  because consolidation is the hardest of these jobs. Also test `gpt-oss:20b` to see whether a
  smaller model is good enough.
- Turn dreaming on with `run_sequence(..., dream_every=n)` (in batches). `n=0` disables it, and
  that is the default.

## Experiment

### Arms

All arms run the same task sequences, seeds and executor. Judge: `CascadeJudge(clef → gpt-oss:20b,
0.3/0.95)`. The four memory arms also repeat with `--ground-truth` to separate judge errors from
memory effects.

| Arm | Write | Between batches | Read |
|---|---|---|---|
| A. none | – | – | – |
| B. JIT (current) | judge gate | – | curator over top-k trajectories |
| C. dream-only | judge gate, store all | dream | notes injected directly |
| D. hybrid | judge gate, store all | dream | curator over notes + top-k |

### Runs

- **Steady layout:** 5 seeds × 40 episodes, `--batch-size 4`, `layout_seed` fixed. That's 200
  episodes per arm.
- **Layout drift:** same as above, but switch `layout_seed` after episode 20. This tests whether
  dreaming retires stale facts (Anthropic's correction finding) and how quickly each arm recovers.
- **Dream cadence (hybrid only):** `dream_every` ∈ {1, 3}.

### Metrics

- **Success rate**, overall and per window of 10 episodes (learning curve).
- **Steps per successful episode.** The kitchen layout is fixed, so good memory should cut search.
- **Briefing "unknown" rate:** how often the briefing says a needed object's location is unknown.
  This tests hypothesis 1 directly.
- **Note accuracy:** the kitchen environment knows the true layout (`make_layout`), so every
  location `fact` can be scored automatically. Report precision and the number of stale facts still
  active after the drift.
- **Rejected dream operations by reason.** This shows how often the model tries to write
  unsupported notes, and therefore how much the evidence check is doing.
- **Cost:** tokens and wall time for curator, dream and judge calls per episode.

Compare arms paired by seed. With binary outcomes and about 200 episodes per arm, report the paired
difference with a bootstrap confidence interval rather than a single number.

### Hypotheses and what would change the default

1. D has a lower "unknown" rate and fewer steps than B on the steady layout.
2. D matches or beats B on success, and beats C (the curator still adds task fit).
3. After the drift, D and C recover more slowly than B unless the dream retires stale facts. Each
   stale fact still active after 2 dreams counts as a failure of the design.
4. The evidence check rejects a meaningful share of proposed operations. If it rejects almost
   none, it isn't earning its keep. If it rejects most, the dream prompt needs work.

Make hybrid the default only if D beats B on steps and is no worse on success on at least 4 of 5
seeds, **and** recovers from the drift within the same number of batches as B or fewer.

## Implementation order

1. Store all trajectories plus verdicts, and `verified_only` retrieval. Tests: old databases still
   open, and retrieval behaviour is unchanged.
2. Notes table, evidence verification and kind rules. Tests on fixed trajectories: a fabricated
   quote is rejected, a quote taken from an action is rejected, and a `procedure` citing a failure
   is rejected.
3. `Dreamer` with a fake LLM (pattern from `tests/conftest.py`), then live.
4. Read path, the dream-only arm and feedback logging.
5. `scripts/eval_hybrid.py`: the arms above with `--save/--load`, like `cascade_sim.py`, and a
   write-up in `docs/evals/`.

## Out of scope

- GRPO training of the curator or dreamer.
- Embedding retrieval. BM25 over notes is enough at 25 notes.
- Wikis or other larger representations. They only make sense with much bigger corpora than the
  kitchen world, and Anthropic found their wiki didn't help on every task.
- MindForge integration.

## Open questions

- Should a `fact` need support from more than one trajectory before the curator sees it? One
  observation is enough for a fixed layout, but maybe not in noisier environments.
- When the environment gives no ground truth, the drift run relies on newer observations
  contradicting old notes. How often does a contradiction never get observed because the executor
  trusts the note and never looks?
- Dreaming happens between batches, so notes are always one batch late. Tasks within a batch can't
  benefit from each other, by design. That's the same limitation Anthropic raises as an open
  question.

## References

- Zhou et al., *Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents*, 2026.
  https://arxiv.org/abs/2609.27334
- Lance Martin, *The components of agent memory*, Anthropic, 6 November 2026.
  https://www.anthropic.com/research/components-of-agent-memory
- `docs/evals/2026-10-03-judges.md` for the judge and cascade used here.
