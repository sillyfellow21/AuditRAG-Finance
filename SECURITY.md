# Security Policy

## Supported Versions
This repository currently supports the latest main branch.

## Reporting a Vulnerability
If you discover a security issue:
1. Do not open a public issue with exploit details.
2. Report privately to the maintainers.
3. Include:
   - impact summary
   - reproduction steps
   - affected files/endpoints
   - suggested mitigation (if known)

## Security Best Practices for This Project
1. Keep API keys only in `.env` and never commit them.
2. Rotate keys immediately if exposed.
3. Review `data/audit_log.jsonl` for suspicious behavior.
4. Keep dependencies updated.
5. Use least privilege for deployment credentials.
