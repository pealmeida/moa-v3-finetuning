# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in GateSwarm MoA Router, please report it responsibly.

**Do not** open a public GitHub issue for security vulnerabilities.

Instead, please:

1. Email the maintainer directly via GitHub's security advisory feature, or
2. Open a private [GitHub Security Advisory](https://github.com/pealmeida/gateswarm-moa-router/security/advisories/new)

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

## Response Timeline

- **Acknowledgment:** within 48 hours
- **Initial assessment:** within 5 business days
- **Patch release:** depends on severity (critical issues prioritized)

## Scope

This policy applies to:

- `router.py` — production scoring engine
- `train.py` — training pipeline
- `llmfit/` — dataset factory toolkit
- `v32_cascade_weights.json` — pre-trained weights

## Out of Scope

- Third-party model provider APIs (report to respective providers)
- Issues in dependencies (numpy, scipy, scikit-learn) — report upstream

Thank you for helping keep GateSwarm secure.
