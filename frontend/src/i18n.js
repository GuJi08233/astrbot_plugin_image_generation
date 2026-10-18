import { ref } from "vue";
import { getBridge } from "./bridge";

export const locale = ref("zh-CN");
export const isDark = ref(false);
export const revision = ref(0);

export function syncContext(context) {
  if (!context || typeof context !== "object") {
    return;
  }
  if (typeof context.locale === "string" && context.locale) {
    locale.value = context.locale;
  }
  if (typeof context.isDark === "boolean") {
    isDark.value = context.isDark;
  }
  revision.value += 1;
}

/**
 * Translate a page string.
 *
 * @param {string} key Dot path relative to `pages.dashboard`.
 * @param {string} fallback Text used when the plugin i18n has no entry.
 * @param {Record<string, unknown>} vars Values for `{name}` placeholders.
 * @returns {string} The resolved text.
 */
export function t(key, fallback = "", vars = {}) {
  // Touch the revision so templates re-render after a locale/theme update.
  revision.value;
  let text = fallback;
  try {
    const bridge = getBridge();
    if (typeof bridge.t === "function") {
      text = bridge.t(`pages.dashboard.${key}`, fallback) || fallback;
    }
  } catch {
    text = fallback;
  }
  return String(text).replace(/\{(\w+)\}/g, (match, name) =>
    name in vars ? String(vars[name]) : match,
  );
}
