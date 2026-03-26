# Security Policy

## Supported Versions

We release patches for security issues on the latest minor release line.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following:

1. **GitHub Security Advisories** — Use [GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) if enabled on this repository.
2. **Email** — Contact the maintainers at the address listed in the repository **About** section or in `pyproject.toml` under `project.maintainers` (when set).

Include:

- Description of the issue and potential impact
- Steps to reproduce (proof of concept if possible)
- Affected versions or commit hashes
- Any suggested fix (optional)

You should receive an initial response within **7 days**. If the issue is confirmed, we will coordinate a fix, release, and public disclosure timeline with you.

## Scope

In scope:

- This repository’s Python code (gateway, providers, API layer)
- Default configuration and documented behavior

Out of scope:

- Third-party LLM provider APIs (report to OpenAI, Anthropic, Google, etc.)
- Misconfiguration by operators (exposed `.env`, disabled security flags) — document in README instead

## Safe Defaults

The gateway ships with:

- `ALLOW_DYNAMIC_RUNTIME_REGISTRATION=true` — operators may lock this down in production
- `ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=false` — BYOK in request bodies is opt-in because keys in JSON are easy to log

Review [README.md](README.md) and [.env.example](.env.example) before deploying publicly.
