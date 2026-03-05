Generate a conventional commit message (feat/fix/chore/docs) for all staged changes.

Rules:
- Subject line: under 72 characters, imperative mood (e.g. "Add", "Fix", "Update")
- Body: bullet list of key files changed and what changed in each
- Footer: Co-Authored-By line

Steps:
1. Run `git diff --staged --stat` to see which files changed
2. Run `git diff --staged` to read the actual diffs
3. Compose the commit message following conventional commit format
4. Output ONLY the commit message text — no explanation, no markdown fences
