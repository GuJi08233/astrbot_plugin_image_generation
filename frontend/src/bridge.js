/**
 * Bridge access layer for the AstrBot plugin page.
 *
 * The page runs inside a sandboxed iframe without `allow-same-origin`, so every
 * request has to go through the injected `AstrBotPluginView` SDK. The local
 * fallback below only exists so the page can be opened directly during
 * development.
 */

let cachedBridge = null;

function createDevFallbackBridge() {
  const apiBase = "/api/plugin/astrbot_plugin_image_generation/";

  const parse = async (response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  };

  return {
    ready: async () => ({}),
    getContext: () => ({}),
    onContext: () => () => {},
    t: (key, fallback) => fallback ?? key,
    apiGet: async (endpoint, params = {}) => {
      const url = new URL(`${apiBase}${endpoint}`, window.location.origin);
      Object.entries(params || {}).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") {
          url.searchParams.set(key, value);
        }
      });
      return parse(await fetch(url, { credentials: "same-origin" }));
    },
    apiPost: async (endpoint, body = {}) =>
      parse(
        await fetch(`${apiBase}${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify(body),
        }),
      ),
    upload: async (endpoint, file) => {
      const form = new FormData();
      form.append("file", file);
      return parse(
        await fetch(`${apiBase}${endpoint}`, {
          method: "POST",
          body: form,
          credentials: "same-origin",
        }),
      );
    },
    download: async (endpoint, params = {}, filename = "image.png") => {
      const url = new URL(`${apiBase}${endpoint}`, window.location.origin);
      Object.entries(params || {}).forEach(([key, value]) => {
        url.searchParams.set(key, String(value));
      });
      const response = await fetch(url, { credentials: "same-origin" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const blobUrl = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = blobUrl;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(blobUrl);
    },
    subscribeSSE: async () => "",
    unsubscribeSSE: async () => undefined,
  };
}

export function getBridge() {
  if (window.AstrBotPluginView) {
    if (cachedBridge !== window.AstrBotPluginView) {
      cachedBridge = window.AstrBotPluginView;
    }
    return cachedBridge;
  }
  if (!cachedBridge) {
    cachedBridge = createDevFallbackBridge();
  }
  return cachedBridge;
}

export async function waitForBridge() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (window.AstrBotPluginView) {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  const bridge = getBridge();
  try {
    await bridge.ready();
  } catch {
    // A ready() failure only means the dashboard did not answer in time.
  }
  return bridge;
}

/** Recover once if a dev fallback instance was captured before injection. */
export function refreshBridge() {
  if (window.AstrBotPluginView) {
    cachedBridge = window.AstrBotPluginView;
  }
  return getBridge();
}
