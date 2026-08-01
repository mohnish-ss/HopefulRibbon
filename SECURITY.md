# Security policy and repository findings

## Reporting

Do not open a public issue containing a credential, private health value, or other sensitive data. Contact the repository owner privately with the affected path, commit, and a minimal reproduction.

## Historical credential exposure

Repository history contains a hard-coded Google Maps API key and a hard-coded email application password from the original implementation. Removing those strings from the current worktree does **not** invalidate credentials or erase Git history.

The owner should:

1. Revoke and rotate both historical credentials immediately.
2. Review provider access and billing logs for unexpected use.
3. Apply API, origin, account, and least-privilege restrictions to replacements.
4. Store replacements only in local/deployment environment variables.
5. Consider a coordinated Git-history rewrite only after rotation, with all collaborators informed.

No automatic history rewrite is performed by this refactor.

## Runtime safeguards

The app requires an environment-provided Flask secret, enforces CSRF protection and a request-size limit, validates model inputs, does not log submitted measurements, bounds external request time, and returns generic failures to users. See `.env.example` for configuration without credentials.
