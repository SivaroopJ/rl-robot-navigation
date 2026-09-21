## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Project vault (Obsidian)

Tracking notes live in a separate Obsidian vault **outside this repo**: `/home/bt3/23CS10067/projects/rl-robot-navigation-vault/`. See its `Vault Guide.md`. The vault is a git repo synced to a private GitHub repo; the user reads it in Obsidian on a Windows laptop. Its `reports/` folder is a copy of this repo's reports that `sync.sh` rebuilds — edit reports here in the repo, never in the vault.

- **Session start:** read the vault's `Home.md`, `Open Threads.md` and the newest file in `journal/`.
- **Before running a new experiment:** create `experiments/<slug>.md` in the vault from `_templates/Experiment.md`, with the gate written before the run.
- **Session end, or after a commit:** append to the vault's `journal/YYYY-MM-DD.md` (template: `_templates/Session Log.md`). Update `Open Threads`, `Timeline`, `Reports Index` and the Home "Where things stand" block when they change. Then run `./sync.sh "<message>"` in the vault to copy in the reports, commit and push (the user pulls on the laptop).
- Link reports with `[[FILENAME]]` wikilinks; don't copy result tables into notes — reports in `MD_files/` stay the source of truth.
