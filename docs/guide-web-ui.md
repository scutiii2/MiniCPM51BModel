# MiniCPM51BModel: the admin web UI

One page, `/admin/`, for tokens and Ollama CPU threads. It is the same for every way of running the service (Docker
on the same device, Docker on a deployment device, no Docker). Only the address differs:

| Run | Address |
|---|---|
| Docker or `run.bat` on this PC | `http://127.0.0.1:9130/admin/` |
| Docker on the deployment box | `http://<box-ip>:9130/admin/` |

If `MINICPM_PORT` is changed, use that port.

## Before you open it
The page needs the admin password: the value of `MINICPM_ADMIN_PASSWORD`.
- Docker: set it in `.env` next to `docker-compose.yml`, then `docker compose up -d`.
- No Docker: set it in the environment before starting the service (`run.bat` / `uvicorn`); the service reads the
  real environment, not `.env`.
- Empty password = no auth at all and the page answers `Admin password required (set MINICPM_ADMIN_PASSWORD)`. Set
  one to use the page.

## Sign in
Type the admin password and press **Load**. A wrong password shows `Wrong admin password`. The password is held in
the page only: reload the tab and sign in again.

## Tokens
Each calling project gets its own token, so one can be revoked without touching the others.
- **Create:** type a project name (up to 64 characters) and press **Create token**. The token appears once as
  `Copy now, it is not shown again: ...`. Copy it before you do anything else.
- **Use:** the caller sends it in the `X-Internal-Token` header on every call except `/health`.
- **List:** the table shows project name and creation time. Token values are not shown again.
- **Revoke:** press **Revoke** on the row and confirm. The caller gets `401 Invalid token` from then on. Lost
  token = revoke it and create a new one.

Tokens live in `tokens.json` (`MINICPM_TOKENS_FILE`). In Docker that is the `tokens` volume at `/data`, so they
survive `docker compose down` and rebuilds. Without Docker it is `tokens.json` in the folder the service starts from.
Tokens are enforced only while `MINICPM_ADMIN_PASSWORD` is set.

## Ollama CPU threads
Field **Ollama CPU threads**, then **Apply**.
- Sets `num_thread` sent to Ollama with each request: how many CPU threads it uses per generation. Start near the
  number of physical cores the host can spare.
- Range 1 to 256. The field shows the value in use now (`num_thread` in `/health`; empty = Ollama decides).
- The change applies at once to the replica that answers the click and is saved to `settings.json` next to
  `tokens.json`. Other replicas pick it up when they restart: `docker compose restart minicpm`.
- The saved value beats `MINICPM_NUM_THREAD` from `.env`. Remove `settings.json` from the `/data` volume to go back
  to the `.env` value.

## Replicas and parallel requests (read-only)
The line **API replicas: N. Requests Ollama runs at once: M** shows `MINICPM_REPLICAS` and `OLLAMA_NUM_PARALLEL`. The
page cannot change them. To change, set them in `.env`, then:
```bash
docker compose up -d --build
docker compose restart minicpm_lb
```
`unknown` means the variable is not set, which is normal without Docker. Without Docker, Ollama's parallelism is
set on the Ollama process itself (`OLLAMA_NUM_PARALLEL` in its environment).

## Problems
| Symptom | Fix |
|---|---|
| Page does not load | Service not running, wrong port, or blocked by a firewall. Test `/health`. |
| `Wrong admin password` | Password differs from `MINICPM_ADMIN_PASSWORD` in the running container. Recreate it after editing `.env`. |
| `Admin password required (set MINICPM_ADMIN_PASSWORD)` | The password is empty. Set it and restart. |
| Token lost | Revoke it and create a new one. |
| `/health` shows `ready: false` | Ollama is up but the model is not pulled yet. See the run guide for `ollama pull`. |
| Never expose port 9130 to the internet | It is a LAN tool; use a firewall or VPN for remote access. |
