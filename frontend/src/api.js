import { getBridge, refreshBridge } from "./bridge";

async function request(method, endpoint, payload) {
  const bridge = getBridge();
  const send = method === "get" ? bridge.apiGet : bridge.apiPost;
  // Vue reactive proxies cannot be structured-cloned by postMessage, so every
  // payload is reduced to plain JSON data before crossing the bridge.
  const plainPayload =
    payload === undefined ? {} : JSON.parse(JSON.stringify(payload));
  try {
    const result = await send.call(bridge, endpoint, plainPayload);
    if (result?.status === "error") {
      throw new Error(result.message || "Request failed");
    }
    return result;
  } catch (error) {
    // A dev fallback may have been captured before the SDK finished injecting.
    const upgraded = refreshBridge();
    if (upgraded === bridge) {
      throw error;
    }
    const result = await (method === "get" ? upgraded.apiGet : upgraded.apiPost).call(
      upgraded,
      endpoint,
      plainPayload,
    );
    if (result?.status === "error") {
      throw new Error(result.message || "Request failed");
    }
    return result;
  }
}

export const apiGet = (endpoint, params = {}) => request("get", endpoint, params);
export const apiPost = (endpoint, body = {}) => request("post", endpoint, body);

export async function uploadFile(endpoint, file) {
  const bridge = getBridge();
  const result = await bridge.upload(endpoint, file);
  if (result?.status === "error") {
    throw new Error(result.message || "Upload failed");
  }
  return result;
}

export async function downloadFile(endpoint, filename) {
  const bridge = getBridge();
  await bridge.download(endpoint, {}, filename);
}

export const taskImagePath = (taskId, index, kind = "preview") =>
  `page/tasks/${encodeURIComponent(taskId)}/images/${index}/${kind}`;
