import { reactive } from "vue";
import { apiGet, taskImagePath } from "./api";

const cache = reactive({});

export function previewKey(taskId, index) {
  return `${taskId}:${index}`;
}

export function usePreview() {
  /**
   * Load one generated image preview through the bridge and cache it.
   *
   * @param {string} taskId Generation task id.
   * @param {number} index 1-based image index.
   * @returns {Promise<{data_url?: string, error?: string, loading?: boolean}>}
   */
  async function loadPreview(taskId, index) {
    const key = previewKey(taskId, index);
    if (cache[key] && !cache[key].loading) {
      return cache[key];
    }
    cache[key] = { loading: true };
    try {
      const result = await apiGet(taskImagePath(taskId, index, "preview"));
      cache[key] = { data_url: result?.data_url || "" };
    } catch (error) {
      cache[key] = { error: error?.message || "preview failed" };
    }
    return cache[key];
  }

  return { cache, loadPreview };
}
