# Security Policy

## Reporting a Vulnerability

Please report any **critical** or **important** security vulnerability, suspected or confirmed, through private disclosure channels:

### Preferred: GitHub Security Advisories

1. Go to the repository's **Security tab**
2. Click **"Report a vulnerability"**
3. Submit the advisory

This creates a private report visible only to maintainers.

### Alternative: Email

If GitHub advisories are not suitable, please contact this subgroup of GridFM maintainers:

- [Romeo Kienzler](mailto:Romeo.Kienzler1@ibm.com)
- [Alban Puech](mailto:Alban.Puech2@ibm.com)
- [Tamara Govindasamy](mailto:tamara.govindasamy@ibm.com)
- [François Mirallès](mailto:miralles.francois@hydroquebec.com)
- [Thomas Tolhurst](mailto:tolhurst.thomas@hydroquebec.com)

---

In your report, include:
- Who you are (name and company)
- Description of the issue
- Affected versions
- Detailed steps to reproduce
- Potential impact
- Suggested remediation (optional)

For **moderate** or **low-severity** security vulnerabilities, you can use public GitHub issues.

To help you assess the severity of the potential vulnerability, you can use the [Apache severity rating](https://security.apache.org/blog/severityrating/).

If you are not sure whether the issue should be reported privately or publicly, please make a private report.

---

## Supported Versions

We currently provide security updates for the following versions:

| Version        | Supported |
|----------------|-----------|
| Latest release | ✅        |
| Previous major | ❌        |
| Older versions | ❌        |

Users are strongly encouraged to upgrade to the latest release to receive security fixes.

---

## Response Timeline

We aim to follow these response targets:

- **Initial acknowledgment**: within 72 hours
- **Status update**: within 7 days
- **Resolution target**: within 90 days (depending on severity)

These are targets, not guarantees.

### Severity Guidelines

| Severity | Response Target | Patch Target |
|----------|----------------|--------------|
| Critical | 24–48 hours    | ≤ 7 days     |
| High     | ≤ 72 hours     | ≤ 14 days    |
| Medium   | ≤ 7 days       | ≤ 30 days    |
| Low      | ≤ 14 days      | ≤ 90 days    |

---

## Disclosure Policy

We follow a **coordinated vulnerability disclosure (CVD)** process:

- We work with reporters to agree on a disclosure timeline
- Public disclosure occurs after a fix is available or mitigation exists
- Contributors are credited unless anonymity is requested
- CVE identifiers will be requested when appropriate

---

## Security Practices

We strive to follow secure software development practices aligned with OpenSSF recommendations:

- Dependency scanning and updates (e.g., Dependabot/Renovate)
- Static analysis (e.g., CodeQL or equivalent)
- Reproducible builds where possible
- Code review before merging
- Use of CI pipelines for validation

---

## Supply Chain Security

Where applicable, we aim to:

- Provide versioned releases with changelogs
- Track dependencies and vulnerabilities
- Improve build provenance over time (e.g., SLSA alignment)

---

## Reporting Abuse or Misuse

If you believe the software is being used in a way that creates security risks or violates acceptable practices, please report it via the same channels above.

---

## Acknowledgements

We thank security researchers and contributors who help improve the safety and reliability of the GridFM ecosystem.
