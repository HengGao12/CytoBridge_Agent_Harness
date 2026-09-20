## Skill-Creator Auto Prompt

Use `$skill-creator` to design or update skills for this request.

### Goal and Trigger Scenario
- Goal: convert requested downstream workflow logic into reusable skill(s) with clear trigger descriptions.
- Trigger: user asks to create/refactor/migrate skill workflows or automate skill creation.

### Scope
- In scope:
  - Skill folder structure (`SKILL.md`, optional `scripts/`, `references/`, `assets/`, `agents/openai.yaml`).
  - Skill naming normalization (hyphen-case).
  - Reusable scripts and references for repeated tasks.
  - Validation with `quick_validate.py`.
- Out of scope:
  - Unrelated model training or biological result generation not needed for skill creation.
  - Extra documentation files such as README/CHANGELOG unless explicitly requested.

### Input/Output Interface
- Input:
  - User raw request: `{user_instruction}`
  - Global goal context: `{user_goal}`
- Output:
  - Concrete skill plan and target folder layout.
  - `SKILL.md` content with strict frontmatter (`name`, `description`).
  - Any required scripts/references/assets and exact file paths.
  - Validation result and fixes if validation fails.

### Resource Suggestions
- `scripts/`: deterministic helpers such as skill scaffolding, lint/validate adapters, or content generation helpers.
- `references/`: style guides, schemas, routing rules, or interface constraints.
- `assets/`: templates/icons/boilerplate files used by generated output.

### Acceptance Criteria
- Generated/updated skills are discoverable by the downstream skill loader.
- Skill names are hyphen-case and unique in domain.
- `scripts/quick_validate.py <skill_dir>` passes.
- Final skill instructions are concise, trigger-aware, and reusable.

