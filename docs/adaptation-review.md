# Evidence-backed adaptation review

Design for reviewing curated prompts, adapted code, configs, and other changed
files against their originals — the experience of reviewing a pull request:
compare original vs. migrated file, explain each meaningful change beside the
diff in clear language with its supporting evidence, let the user accept or
reject individual changes, and keep the migrated artifact itself clean (no
explanatory comments inside deliverables).

Status: all three phases (v1.4.0-a, v1.4.0-b, v1.4.0-c) implemented.
Designed and implemented 2026-09-21.

## Problem

Two gaps motivated this design:

1. **"It still gives the prompt with no adaptation."** The toolkit never
   rewrites prompt text itself (`core/migration.py`: the deterministic
   candidate is the source prompt verbatim; every recommended change is
   carried as evidence-linked advice). The actual rewrite is delegated to the
   host agent during a guided run. Two things made this look broken rather
   than deliberate: the worklist field was named `deterministic_candidate`,
   inviting hosts to present the unmodified input as the tool's proposed
   adaptation; and nothing forced the agent to *reconcile* the guidance list
   against the prompt — every guidance item could be silently ignored, and the
   report could not distinguish "reviewed and genuinely fine" from "nobody
   engaged with the guidance".
2. **Change records are free text.** `AdaptationEntry.changes` is a list of
   unanchored strings. A reviewer cannot see which edit in the file a claim
   refers to, cannot verify the claimed evidence, and cannot accept or reject
   one change independently of the others.

## Desired review experience

For every deliverable under a run's `output/` directory:

- The original and adapted content are compared automatically.
- Each meaningful change is explained beside its diff hunk in the format:
  `<one sentence: why the change was made, e.g. which target-model behavior it
  adapts to>; evidence: <source link / analysis reference>` — evidence is
  omitted only for mechanical fixes (typos, dead whitespace).
- The user accepts or rejects individual changes; rejections deterministically
  regenerate the deliverable from the accepted changes only.
- A deliverable that needs no change is stated explicitly in the report, with
  the reasoning, never left as an unexplained identical file.
- Deliverables stay clean: all annotation lives in `changes.yaml` and the
  review records, never inside the adapted files.

## Design

The design reuses the interactive blocker-resolution pattern (v1.4):
deterministic derivation, the host agent presents records verbatim, user
decisions recorded durably and re-applied, staleness reported rather than
silently absorbed. The toolkit stays deterministic and local-first; the host
agent supplies the semantic work and the toolkit verifies it fail-closed.

### Data model

The unit of record becomes a hunk-anchored, evidence-linked change instead of
a free-text bullet:

```yaml
annotated_changes:
  - id: c1                      # stable within the file's submission
    operation: delete           # edit | insert | delete | restructure
    original_anchor: "You must think step by step..."  # exact span in the original
    adapted_anchor: null        # exact span in the adapted file (null for delete)
    why: >-
      Removed the explicit chain-of-thought paragraph; the target model
      reasons natively and detailed step instructions can constrain it.
    evidence:
      - kind: model_guidance    # model_guidance | model_difference |
                                # analysis_finding | research | mechanical
        url: https://...
        reference: reasoning_guidance   # registry field / report section /
                                        # finding category
guidance_dispositions:          # every guidance item gets a verdict
  - guidance_id: "g:1a2b3c4d5e"
    disposition: applied        # applied | not_applicable | declined
    note: ""                    # required when declined
unchanged: false
```

This extends what already exists rather than inventing a parallel vocabulary:
`MigrationAdvice(text, basis, evidence_urls)` is the evidence-carrying advice
unit, `PromptSemanticChange` already has state/before/after/rationale, and
stable content-derived ids follow the `MigrationBlocker` id scheme so recorded
decisions keep matching across regenerations and go stale when content
changes.

### Submission-time validation (deterministic, fail-closed)

`submit_adapted_prompt` / `submit_adapted_file` compute the real diff over the
decoded runtime values (the same machinery that already rejects
serialization-only edits) and enforce:

- **No undocumented edits**: every diff hunk must be covered by an annotation
  whose anchors resolve in the original/adapted text.
- **No phantom claims**: every annotation must correspond to a real hunk.
- **Evidence honesty**: non-`mechanical` annotations need at least one
  evidence entry; URLs are checked against the run's known evidence set (the
  registry/guidance/research URLs the plan already carries) — unknown URLs
  warn, missing evidence rejects.
- **Guidance reconciliation**: every guidance item on the prompt task (its own
  `guidance` plus the worklist's `shared_prompt_guidance`) must be disposed as
  `applied`, `not_applicable`, or `declined` (with a note). The agent can
  still conclude "no change", but only by explicitly disposing each item, and
  the dispositions appear in the report.

### Review workflow (mirrors blocker resolution)

- `get_change_review(run_dir)` (MCP) / `llm-migrate run review` (CLI): per
  deliverable, the original-to-adapted diff with each hunk paired to its
  annotation (why + evidence) and its status (`pending | accepted |
  rejected`). Because the workspace stores no copy of originals, the diff is
  computed against the live application file and refuses (reports stale) when
  the file no longer matches the entry's recorded `source_sha256`.
- `record_change_decision(run_dir, source_path, change_id, decision, note?)` /
  `llm-migrate run decide-change`: durable in the run's review decision log,
  keyed to the entry's `source_sha256`/`adapted_sha256`; a resubmission marks
  prior decisions for that file stale, never silently applied.
- **Rejection regenerates deterministically**: the deliverable is rebuilt by
  applying only accepted hunks to the original. Rejecting every change reverts
  the deliverable to a reviewed-unchanged entry.
- Each review item reports the verification state of every evidence entry
  (v1.5.2): `registry-recorded`, `plan-carried`, `UNKNOWN — not among this
  run's plan or registry evidence`, `mechanical`, or `reference only`, so
  the accept/reject decision sees what the submission gate saw.
- Guidance whose trigger the application never exhibits is pre-disposed
  `not_applicable` by the toolkit with its reason (v1.5.2, resolved coverage
  only); the log records it and the report marks it PRE-DISPOSED. An
  explicit disposition from the agent still wins.
- `finalize_migration` reports undecided changes as their own bucket alongside
  coverage gaps but does not block on them by default; review may follow
  finalization. A strict run (V1.5) refuses to finalize cleanly while
  undecided changes remain.

### Reporting

- Deliverables reviewed with no change needed render under their own explicit
  heading ("no change needed") with the recorded reasoning and the guidance
  dispositions — never as a generic "what changed" bullet. Finalization counts
  `reviewed_unchanged` separately from adapted deliverables.
- Changed deliverables render each change in the review format — one line
  `<why>; evidence: <url or reference>` beside its diff hunk — plus its
  accept/reject status once decided.

## Non-goals

- **Inline explanatory comments in artifacts** — annotations live beside the
  artifact, never in it.
- **Patch-file-per-change / git plumbing** — anchors in YAML are simpler,
  testable, and consistent with the all-YAML run workspace.
- **A bundled graphical diff viewer** — the host agent presenting each change
  verbatim is the proven UX across hosts; an exported side-by-side HTML view
  can be added later without changing the data model.
- **Automatic application of accepted changes to the application tree** —
  applying deliverables remains an explicit human step.

## Phases

### v1.4.0-a — report honesty and guidance accountability (implemented)

- Rename the worklist field `deterministic_candidate` to `verbatim_source`
  (`AdaptationTaskList` schema version 2) and state in the task guidance that
  it is the unmodified input, never a proposed adaptation. (Schema is
  version 4 today: v3 added file-task `required_changes` ids, v4 the V1.5
  `unaffected_files` bucket.)
- Give every prompt-task guidance item a stable content-derived id
  (`GuidanceItem`), covering both per-task `guidance` and
  `shared_prompt_guidance`.
- Require `guidance_dispositions` on every prompt submission: each guidance
  id disposed exactly once as `applied` / `not_applicable` / `declined`
  (note required when declined); an `unchanged=true` submission cannot carry
  an `applied` disposition. Missing, unknown, duplicate, or contradictory
  dispositions reject the submission with the exact ids and texts still
  owed.
- Record `unchanged` and the dispositions on `AdaptationEntry`
  (`changes.yaml`), render reviewed-unchanged deliverables under an explicit
  "no change needed" heading with their reasoning and dispositions, and
  count `reviewed_unchanged` separately in `MigrationRunFinalization`
  (schema version 3 then; version 5 today — v4 added the undecided-changes
  bucket, v5 the V1.5 consistency findings, unconfirmed unaffected files,
  validation disposition, and strict violations).

### v1.4.0-b — annotated, hunk-anchored changes (implemented)

- `AnnotatedChange` / `ChangeEvidence` models (`core/annotations.py`):
  operation (`edit` / `insert` / `delete` / `restructure`), exact-span
  anchors, one-sentence `why`, evidence entries, and a content-derived
  stable id filled at acceptance; `changes.yaml` schema version 2 (version 1
  logs still load).
- A `new_file` submission must carry at least one evidence-linked annotated
  change (v1.5.1): an anchor-less insert/restructure claims the whole file,
  and without annotations there is nothing for change review to decide.
- Diff computation over the decoded runtime values in both submit paths
  (structured prompt documents decode to their component values, so
  serialization tricks change nothing): every hunk must be covered by an
  annotation whose anchors resolve, every annotation must match a real hunk,
  non-`mechanical` evidence needs a url or reference, and evidence URLs are
  checked against the plan's known evidence set (unknown URLs warn). A
  `restructure` annotation with no anchors claims the whole rewrite. An
  `unchanged=true` submission cannot carry annotations.
- File tasks gained disposition parity: `required_changes` carry stable ids
  and file submissions dispose them like prompt guidance.
- The report renders each change as its why-plus-evidence line with the
  before/after anchored spans; the deliverable files stay clean.
- Surfaces: CLI `--annotations <yaml file>` on `run submit-prompt` /
  `run submit-file` (plus `--dispose` on `submit-file`), MCP
  `annotated_changes` / `guidance_dispositions` parameters on both
  submission tools.

### v1.4.0-c — change review decisions (implemented)

- `get_change_review` / `record_change_decision` (MCP) and `run review` /
  `run decide-change` (CLI, `core/change_review.py`): per deliverable, the
  decoded unified diff plus every annotated change with its why, evidence,
  before/after spans, and status (pending / accepted / rejected), with
  verbatim-presentation guidance — the user decides, never the agent.
- Durable decisions in the run's `change-decisions.yaml`, keyed to the
  submission fingerprints: a resubmission makes them stale (reported, never
  silently applied), a log copied from another run is refused, and a review
  whose application file drifted is frozen until the adaptation is re-run.
- Deterministic regeneration on every decision: submissions keep an
  as-submitted copy under `review/submissions/`, and the deliverable under
  `output/` is rebuilt from (original, submission, live rejections) —
  rejected regions revert to the original, everything else keeps the
  submission, structured documents are rebuilt per component with non-prompt
  values preserved, and a decision whose rejections cannot be applied
  deterministically (mixed-coverage regions, unmappable anchors, message
  edits, TOML) is refused rather than recorded. Because the submission is
  preserved, decisions can be re-made in any order. Rejecting every change
  leaves no annotated adaptation in the deliverable.
- Finalization counts undecided changes as their own bucket
  (`MigrationRunFinalization` schema version 4 at the time) without blocking
  by default, and the
  report renders each change's decision (ACCEPTED / REJECTED with the note /
  pending review) plus a per-file review outcome line.
