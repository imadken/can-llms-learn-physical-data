# RSS 2027: supervised preliminary work

Version 2.2-locked, 20 November 2026. Status: author plan frozen; no experiments executed by the supervisor.

Proposed paper: **What Transfers from Effect-Conditioned Action Pretraining to Vision-Language-Action Policies?**

This package fixes the current research scope and delegates implementation of three exploratory preliminaries to Codex. The supervisor retains scientific decisions and interpretation. Preliminary outcomes support the Stage 1 paper; they do not automatically change its question or expand the experiment campaign.

This final lock incorporates the concrete conditional-action transfer hypothesis, the practitioner's supervision/compute decision, concise Stage 2 reasoning, and a deterministic training-only effect/token calibration rule. The title, primary comparisons, six-dimensional effect, experiment counts and compute ceilings are fixed. The scientific manuscript is developed in parallel with execution; it does not wait for favorable pilot results. The evidence cutoff is 28 November 2026 at 18:00 Europe/Paris.

Verify `RESEARCH_LOCK.json` before execution. It hashes the protected scientific documents, configuration and prompt. Runtime facts and computed calibration constants belong in separate execution artifacts. Outcome-driven amendments are prohibited. Any proposed scientific amendment requires a new explicit user instruction and must preserve this frozen release.

## Read in this order

1. `research/STAGE1_SCOPE_AND_CLAIMS.md`: current scientific specification and the limited changes to version 1.
2. `research/PRELIMINARY_PROTOCOL.md`: the three experiments, fixed quantities, execution limits and audit requirements.
3. `configs/preliminary_locked.json`: machine-readable constants; must agree with the protocol.
4. `prompts/CODEX_PRELIMINARIES.txt`: the execution prompt.
5. `research/REVIEWER_OBJECTIONS.md`: the supervisor's argument and claim boundaries, not a request for Codex to redesign the study.

The archived version 1 plan and its bibliography preserve the earlier complete design. It is historical context. Current files supersede its eight-dimensional effect token, single-primary-contrast framing, mandatory physical shifts/probes, and preliminary sequence. Do not execute its full Stage 2 schedule.

## Repository workflow

The repository URL was not identified in the conversation. This package has not been pushed to a guessed repository. Use the exact repository selected by the user. Do not infer it from another RSS project's name.

Once the user opens the intended checkout for Codex, inspect the remote and any applicable AGENTS.md instructions. Create a dedicated branch `research/effect-pretraining-stage1-v2` from the supplied project base without discarding work. Commit this package before implementation. Commit code, tests, resolved source facts, configuration hashes, small result tables and a concise report there. Preserve full datasets, checkpoints and replay snapshots on the DGX Spark with checksummed paths. Do not place large generated artifacts or credentials in Git.

Use `audit/` for the small review package and create a separate compressed evidence bundle as described in the protocol. Return the repository URL, branch and exact implementation/result commit hashes. The supervisor will review source and numerical evidence before deciding which results enter the paper.

## What is authorized by the execution prompt

Implement and run only the three bounded preliminaries after correctness checks and measured resource profiling. There is no authorization for the full Stage 2 campaign, new experimental conditions, additional task searches, long unattended sweeps, paid services, or changes to the scientific claims. Routine engineering choices that preserve the specified experiment are allowed and must be logged.

An invalid measurement or implementation defect may require a correction. A negative, small, mixed, or unexpectedly favorable scientific result does not authorize redesign or more runs.
