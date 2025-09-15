# monkapp

This project models an intelligent building heating control system. A lightweight API is now exposed so the simulation can be deployed on [Vercel](https://vercel.com/) and queried on demand.

## Vercel deployment

Vercel detects the serverless function placed in the `api/` directory and uses `requirements.txt` to install the Python dependencies. The provided `vercel.json` file pins the runtime to Python 3.11 and redirects requests to the simulation endpoint so visiting the deployment root immediately returns JSON output.

### Deploying

1. Install the Vercel CLI and log in (`npm i -g vercel && vercel login`).
2. From the repository root run `vercel` to create the project and `vercel --prod` to ship a production build.
3. Once deployed, queries are served at `/api/simulate` (the root path also redirects there).

### API usage

The endpoint runs a short thermal simulation and returns a summary with a per-hour timeline. Parameters can be adjusted through query string arguments.

```
GET https://<your-deployment>.vercel.app/api/simulate?hours=48&heating_power=2500&occupancy_gain=200
```

Query parameters:

- `hours` – number of simulation hours (1–168, default 24)
- `heating_power` – constant heating power applied in watts (default 2000)
- `occupancy_gain` – internal heat gains from occupants/appliances in watts (default 150)

The response includes a `summary` block with aggregated indicators and a `timeline` array describing the indoor, wall, and air temperatures for each simulated hour.
