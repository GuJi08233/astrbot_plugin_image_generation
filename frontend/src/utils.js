import { t } from "./i18n";

export const ASPECT_RATIOS = [
  "不指定",
  "1:1",
  "2:3",
  "3:2",
  "3:4",
  "4:3",
  "4:5",
  "5:4",
  "9:16",
  "16:9",
  "21:9",
];

export const RESOLUTIONS = ["不指定", "1K", "1.5K", "2K", "4K"];

const STATUS_COLORS = {
  queued: "grey",
  pending: "grey",
  running: "primary",
  cancelling: "warning",
  succeeded: "success",
  failed: "error",
  cancelled: "grey-darken-1",
};

const STATUS_FALLBACKS = {
  queued: "排队中",
  pending: "等待中",
  running: "生成中",
  cancelling: "取消中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

export const statusColor = (status) => STATUS_COLORS[status] || "grey";

const STATUS_CSS_COLORS = {
  queued: "rgb(var(--v-theme-secondary))",
  pending: "rgb(var(--v-theme-secondary))",
  running: "rgb(var(--v-theme-primary))",
  cancelling: "rgb(var(--v-theme-warning))",
  succeeded: "rgb(var(--v-theme-success))",
  failed: "rgb(var(--v-theme-error))",
  cancelled: "rgba(var(--v-theme-on-surface), 0.45)",
};

export const statusCssColor = (status) =>
  STATUS_CSS_COLORS[status] || "rgba(var(--v-theme-on-surface), 0.45)";

export const statusLabel = (status) =>
  t(`status.${status}`, STATUS_FALLBACKS[status] || status || "unknown");

export function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) {
    return "--";
  }
  if (value < 60) {
    return `${value.toFixed(value < 10 ? 2 : 1)}s`;
  }
  const total = Math.round(value);
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${minutes}m ${rest}s`;
}

export function formatDateTime(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  const pad = (part) => String(part).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function formatBytes(size) {
  const value = Number(size);
  if (!Number.isFinite(value) || value <= 0) {
    return "--";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
