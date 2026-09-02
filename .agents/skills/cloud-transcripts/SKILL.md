---
name: cloud-transcripts
description: >-
  Fetch and search Cursor Cloud Agent chat transcripts via the Cloud Agents API.
  Use when the user asks about prior cloud agent conversations, mess3_feedback_cycle,
  cloud agent history, or when local agent-transcripts are missing cloud-only work.
---

# Cloud agent transcripts

Local chats land in `~/.cursor/projects/.../agent-transcripts/`. Pure Cloud Agent
sessions do not. This skill pulls them through the Cloud Agents API and caches
them under `.cursor/cloud-transcripts/` in a JSONL shape local agents can read.

## Prerequisites

1. Create a user API key at [Cursor Dashboard → API Keys](https://cursor.com/dashboard/api).
2. Add it to `.env.local` in the repo root:

   ```bash
   export CURSOR_API_KEY="key_..."
   ```

3. Load secrets before running commands:

   ```bash
   source scripts/load_env.sh
   ```

Never commit the key. `.env.local` is already gitignored.

Optional sanity check:

```bash
curl --request GET --url https://api.cursor.com/v0/me -u "$CURSOR_API_KEY:"
```

## Fetch workflow

Run from the repo root with `uv run python` or plain `python3`:

```bash
source scripts/load_env.sh

# Recent cloud agents (id, status, createdAt, name, repo, pr)
uv run python .cursor/skills/cloud-transcripts/scripts/fetch_cloud_transcripts.py list --limit 30

# Cache the last 20 conversations locally
uv run python .cursor/skills/cloud-transcripts/scripts/fetch_cloud_transcripts.py fetch-recent --limit 20

# Fetch one agent by id (bc_... or bc-...)
uv run python .cursor/skills/cloud-transcripts/scripts/fetch_cloud_transcripts.py fetch bc_abc123

# Search cached transcripts
uv run python .cursor/skills/cloud-transcripts/scripts/fetch_cloud_transcripts.py search mess3_feedback
```

Outputs:

- `.cursor/cloud-transcripts/{agent_id}.jsonl` — conversation in local-transcript-like JSONL
- `.cursor/cloud-transcripts/{agent_id}.meta.json` — repo, branch, PR, summary, timestamps
- `.cursor/cloud-transcripts/index.json` — manifest of fetched agents

Read those files directly when answering questions about past cloud work.

## When the user asks about a cloud study

1. Search the cache first:

   ```bash
   uv run python .cursor/skills/cloud-transcripts/scripts/fetch_cloud_transcripts.py search "<topic>"
   rg -i "<topic>" .cursor/cloud-transcripts/
   ```

2. If nothing matches, refresh recent agents and search again:

   ```bash
   source scripts/load_env.sh
   uv run python .cursor/skills/cloud-transcripts/scripts/fetch_cloud_transcripts.py fetch-recent --limit 50
   ```

3. If you know the PR, list agents and pick the matching `prUrl`:

   ```bash
   uv run python .cursor/skills/cloud-transcripts/scripts/fetch_cloud_transcripts.py list --limit 50
   ```

4. Fall back to git evidence (`cursor/*` branches, merged PRs) when the agent
   was deleted — deleted cloud agents lose API conversation access.

## Limits

- Conversation fetch uses the legacy v0 endpoint
  `GET /v0/agents/{id}/conversation` (user + assistant text only; no tool calls).
- Agent listing prefers v1, then falls back to v0.
- Deleted cloud agents return errors; archive instead of delete when possible.
- Cached transcripts may contain secrets from prompts — keep them local
  (`.cursor/cloud-transcripts/` is gitignored).

## API reference

- [Cloud Agents API](https://cursor.com/docs/cloud-agent/api/endpoints)
- [v0 conversation endpoint](https://cursor.com/docs/cloud-agent/api/v0.md)
