# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `main` | Yes |
| Older releases | Best effort |

## Report a vulnerability privately

Do not open a public issue for suspected vulnerabilities.

Use GitHub's private vulnerability reporting form:

<https://github.com/Athenaxlee/llm-migrate/security/advisories/new>

Include, when possible:

- affected version or commit
- impact and realistic attack scenario
- minimal reproduction or proof of concept
- affected files or tools
- suggested mitigation
- whether the issue is already public

The maintainer will make a best-effort acknowledgement within seven days,
validate the report, coordinate a fix and disclosure, and credit the reporter
unless anonymity is requested.

## Security-sensitive areas

Reports are especially useful for:

- credential or secret exposure in artifacts, logs, errors, or subprocesses
- unsafe file writes, path traversal, or source overwrite protection bypasses
- prompt injection that changes orchestration authority or tool permissions
- session-registry tampering, hash bypasses, or trust-level escalation
- source-refetch SSRF, redirect, or network-boundary bypasses
- unsafe evaluator or plugin loading
- untrusted YAML, JSON Schema, or repository-content handling

Please do not include real credentials, customer data, or private repository
contents in a report.
