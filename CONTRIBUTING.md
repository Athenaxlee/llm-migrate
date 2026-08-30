# Contributing to llm-migrate

Thank you for helping make LLM migrations safer and more reviewable.

## Start here

- Use GitHub Discussions for questions and early design exploration.
- Search existing issues before opening a new one.
- Use the provided issue forms for bugs, features, and registry updates.
- Report security problems privately according to `SECURITY.md`.
- Keep pull requests focused. Large architectural changes should start with an
  issue or discussion.

## Development setup

```bash
git clone https://github.com/Athenaxlee/llm-migrate.git
cd llm-migrate

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the required checks:

```bash
pytest
ruff check .
ruff format --check .
mypy
python -m pip wheel --no-deps . --wheel-dir dist
```

The default test suite must not require network access, provider credentials, or
paid inference.

## Contribution workflow

1. Fork the repository.
2. Create a branch from current `main`.
3. Add or update tests with the change.
4. Run the required checks locally.
5. Open a pull request using the template.
6. Address review and CI feedback without force-pushing over another
   contributor's work.

Maintainers use squash merges so each pull request becomes one reviewable commit
on `main`.

## Design expectations

Contributions must preserve these boundaries:

- local-first and provider-neutral core behavior
- one shared service layer behind MCP and CLI
- explicit separation of model, provider, platform, and endpoint identity
- deterministic handling wherever semantic inference is unnecessary
- review artifacts before source mutation
- no project-owned provider credentials or hosted inference dependency
- explicit unknowns and blockers instead of guessed compatibility

Read `docs/architecture.md` and `docs/project_context.md` before changing public
contracts or data flow.

## Registry and evidence changes

The registry is a trust boundary. Do not edit canonical model records from an
unreviewed model response or search result.

Registry changes must follow:

```text
research
  -> ResearchResult
  -> propose_registry_update(...)
  -> proposal + evidence + candidate artifacts
  -> maintainer review
  -> canonical registry
```

A registry pull request should include:

- authoritative sources appropriate to each fact topic
- retrieval dates and provider/platform scope
- claim-level evidence and unresolved conflicts
- a generated proposal bundle
- the complete candidate profile
- registry, proposal, schema-parity, and relevant migration tests

Follow `docs/research-policy.md`. Third-party pricing must stay scoped to its
source and must not be presented as direct-provider pricing.

## Tests and artifacts

- Add unit tests for new deterministic rules and failure modes.
- Use fixture repositories for scanner behavior.
- Add golden or snapshot tests for stable plans, reports, proposals, and
  research-session artifacts.
- Never update goldens merely to silence a failure; review the semantic diff.
- Tests must use fake credentials and mocked or recorded provider responses.
- Never commit API keys, private repositories, production prompts, customer
  data, full webpage archives, or model private reasoning.

## AI-assisted contributions

AI-assisted work is welcome, but the human contributor remains responsible for
the code, evidence, tests, licensing, and security of the submission.

- Disclose material AI assistance in the pull request.
- Review generated code and citations before submitting.
- Do not send repository secrets or private data to external models.
- Do not use an agent's agreement as evidence for a registry fact.
- Keep researcher and evidence-review roles independent.

Low-effort generated pull requests that do not demonstrate understanding,
testing, or evidence quality may be closed.

## Documentation

Update documentation when a change affects installation, public commands, MCP
tools, schemas, security behavior, supported platforms, or durable architecture.
Keep the README focused on first-time use; put detailed contracts in `docs/`.

## Licensing

By submitting a contribution, you agree that it is your original work or that
you have the right to submit it, and that it is licensed under Apache-2.0 with
the rest of the project.

All contributions require review and passing CI. Maintainers may request changes
or decline work that conflicts with the project's scope, safety model, or
evidence standards.
