# AI Trader

Monorepo for crypto ML trading research.

## How to Work Here

### Entry Point

For any feature that needs 3+ steps: say **"plan and build X"**. This triggers the `plan-and-build` skill.

For everything else: just do it.

### Pipeline (plan-and-build only)

```
1. Grill     ← you talk, agent asks questions
2. PRD       ← agent creates GitHub issue
3. Issues    ← agent breaks into tasks
4. Execute   ← subagents implement each task
5. PR        ← agent creates PR, code review
6. STOP      ← you test locally, approve or request changes
```

### Rules

1. **Load skills before using them.** Before any phase, load the skill:
   - Grill → load `grill-with-docs`
   - PRD → load `to-prd`
   - Issues → load `to-issues`
   - Implementation → load `tdd`
   - Review → load `review`

2. **Update GitHub labels as you go:**
   - Issue created → `ready-for-agent` + `AFK`
   - Subagent starts → remove both, add `in-progress`
   - Subagent done → remove `in-progress`, add `completed`, close issue

3. **Stop at the human gate.** After PR + code review, present test instructions and WAIT.

4. **Run `make lint` and `make test` before pushing.**

5. **Never push if CI would fail.**

### CI/CD

GitHub Actions blocks merge if tests or lint fail. No exceptions.

```bash
make test    # run tests
make lint    # check linting
make format  # auto-fix formatting
```

### Project Structure

```
ai-trader/
├── projects/
│   └── scraper/     # WhiteBIT data collector
├── shared/          # (future) shared utilities
├── Makefile
├── pyproject.toml
└── README.md
```

New projects go in `projects/{name}/` with their own `README.md`, `requirements.txt`, `tests/`.

### Labels

| Label | Meaning |
|-------|---------|
| `ready-for-agent` | Task ready to pick up |
| `AFK` | No human needed during execution |
| `in-progress` | Agent working on it |
| `completed` | Done, tests passing |
| `blocked` | Can't proceed |
| `needs-human` | Needs your decision |
