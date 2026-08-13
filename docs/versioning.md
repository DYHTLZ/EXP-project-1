# Versioning & rollback

Every update pushed to GitHub is marked with a version tag
(`v1.0.0`, `v1.1.0`, ...). Tags are immutable snapshots: if a new upload
breaks something, any previous version can be restored.

## Version history

| Version | What changed |
|---|---|
| v1.0.0 | Project skeleton (README, LICENSE, .gitignore, structure) |
| v1.1.0 | Week 1: synthetic data generator, loader, EDA notebook |
| v1.2.0 | Weeks 2–4: pricing baseline (frequency + severity GLMs, pure premiums, holdout validation) |
| v1.3.0 | Week 5: fairness audit layer (parity, equalized odds, calibration, Chouldechova demo) |
| v1.4.0 | TEST: public-data applications (UCI Adult, freMTPL2) |
| v1.5.0 | Weeks 6–10: constrained pricing frontier, redistribution, dashboard, memos |
| v1.6.0 | Enhancement: age splines fix, model card, regulatory mapping, CI, employer-focused README |
| v1.6.1 | CI fix: pytest pythonpath |
| v1.6.2 | Versioning & rollback documentation |
| v1.8.0 | Individual fairness analysis (same-risk premium spread) |
| v1.9.0 | Regulatory "remove variable X" model comparison |

## How to check versions

```bash
git tag -l            # list local tags
git tag -l "v*"       # annotated tags carry messages
```

GitHub also exposes every version as a **Release** (with a downloadable zip
archive) on the repository's Releases page.

## How to restore an old version

**Option A — safe restore (keeps history, recommended):**

```bash
git restore --source=v1.5.0 --staged --worktree .
git commit -m "Restore v1.5.0"
git push
```

This puts the old files back as a new commit; the failed version stays in
history and can be examined later.

**Option B — replace `main` entirely (destructive):**

```bash
git checkout -b restore/v1.5.0 v1.5.0
git push origin restore/v1.5.0:main --force
```

Use only when you intentionally want the old version to *become* main.

**Option C — download:** GitHub Releases → choose the version → download the
zip archive.

## Rule for future updates

Every push to `main` gets a version tag:

```bash
git tag -a v1.7.0 -m "What changed"
git push origin main --tags
```

Bumping convention: **patch** for fixes (`v1.6.1` → `v1.6.2`), **minor** for
features (`v1.6` → `v1.7`), **major** for overhauls (`v1` → `v2`).
