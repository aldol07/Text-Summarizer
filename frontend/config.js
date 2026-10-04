// Where the API lives. Empty string = same origin (local: `uvicorn textSummarizer.api.main:app`).
// For the Vercel deployment, set this to the Render URL, e.g. "https://text-summarizer-api.onrender.com".
window.APP_CONFIG = {
  API_BASE_URL: "",
};
