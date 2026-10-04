// Where the API lives.
// Locally (served by FastAPI) the API is on the same origin, so the base URL is empty.
// Anywhere else (Vercel), requests go to the Render service. Update RENDER_URL if your service name differs.
const RENDER_URL = "https://text-summarizer-api.onrender.com";
const isLocal = ["localhost", "127.0.0.1", ""].includes(window.location.hostname);

window.APP_CONFIG = {
  API_BASE_URL: isLocal ? "" : RENDER_URL,
};
