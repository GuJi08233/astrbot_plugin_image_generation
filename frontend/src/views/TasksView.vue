<template>
  <v-row dense>
    <v-col cols="12" lg="5">
      <v-card variant="outlined" class="rounded-lg d-flex flex-column h-100">
        <div class="card-title-row">
          <div>
            <div class="section-caption">{{ t("tasks.eyebrow", "Queue") }}</div>
            <div class="card-title-text mt-1">{{ t("tasks.title", "任务队列") }}</div>
          </div>
          <v-btn
            variant="tonal"
            size="small"
            prepend-icon="$refresh"
            :loading="loading"
            @click="reload(true)"
          >
            {{ t("refresh", "刷新") }}
          </v-btn>
        </div>
        <v-divider />

        <div class="filter-grid pa-4 pb-2">
          <v-text-field
            v-model="filters.keyword"
            :placeholder="t('tasks.keywordPlaceholder', '任务 ID / 提示词 / 预设 / 人设 / 模型 / UMO')"
            density="compact"
            variant="outlined"
            hide-details
            clearable
            prepend-inner-icon="$image"
            class="filter-keyword"
            @update:model-value="debouncedReload"
          />
          <v-select
            v-model="filters.status"
            :items="statusOptions"
            density="compact"
            variant="outlined"
            hide-details
            @update:model-value="reload(true)"
          />
          <v-select
            v-model="filters.source"
            :items="sourceOptions"
            density="compact"
            variant="outlined"
            hide-details
            @update:model-value="reload(true)"
          />
          <v-select
            v-model="filters.model"
            :items="modelOptions"
            density="compact"
            variant="outlined"
            hide-details
            @update:model-value="reload(true)"
          />
          <v-select
            v-model="filters.days"
            :items="dayOptions"
            density="compact"
            variant="outlined"
            hide-details
            @update:model-value="reload(true)"
          />
          <v-select
            v-model="filters.limit"
            :items="[10, 30, 50, 100]"
            density="compact"
            variant="outlined"
            hide-details
            @update:model-value="reload(true)"
          />
          <div class="filter-count">
            {{ t("tasks.count", "共 {total} 条，当前第 {page}/{pages} 页", countText) }}
          </div>
        </div>

        <v-card-text class="flex-grow-1 pa-0 task-list">
          <div v-if="!tasks.length" class="list-empty">
            {{ t("tasks.empty", "暂无任务") }}
          </div>
          <div v-else class="task-rows">
            <button
              v-for="task in tasks"
              :key="task.task_id"
              type="button"
              class="task-row"
              :class="{ 'task-row--active': task.task_id === selectedId }"
              :style="{ '--status-color': statusCssColor(task.status) }"
              @click="selectTask(task.task_id)"
            >
              <span
                class="task-dot"
                :style="{ background: statusCssColor(task.status) }"
              />
              <span class="task-row__main">
                <span class="task-row__title">
                  {{ task.prompt_summary || task.task_id }}
                </span>
                <span class="task-row__meta">
                  <span class="mono">{{ task.task_id }}</span>
                  <span class="dot-sep">·</span>
                  <span>{{ task.model || "--" }}</span>
                  <span class="dot-sep">·</span>
                  <span>{{ formatDateTime(task.created_at) }}</span>
                </span>
              </span>
              <span class="task-row__side">
                <span class="task-row__count">
                  {{ task.result_count || 0 }}/{{ task.requested_count || 0 }}
                </span>
                <span
                  class="task-row__status"
                  :style="{ color: statusCssColor(task.status) }"
                >
                  {{ statusLabel(task.status) }}
                </span>
              </span>
            </button>
          </div>
        </v-card-text>

        <v-divider />
        <div class="d-flex align-center px-2 py-1">
          <v-btn
            variant="text"
            size="small"
            :disabled="offset <= 0"
            @click="reload(false, -1)"
          >
            {{ t("tasks.prevPage", "上一页") }}
          </v-btn>
          <v-spacer />
          <span class="section-caption">
            {{ t("tasks.pageInfo", "第 {page} / {pages} 页", countText) }}
          </span>
          <v-spacer />
          <v-btn
            variant="text"
            size="small"
            :disabled="offset + filters.limit >= total"
            @click="reload(false, 1)"
          >
            {{ t("tasks.nextPage", "下一页") }}
          </v-btn>
        </div>
      </v-card>
    </v-col>

    <v-col cols="12" lg="7">
      <v-card variant="outlined" class="rounded-lg h-100">
        <div class="card-title-row">
          <span class="card-title-text">{{ t("detail.title", "任务详情") }}</span>
          <div v-if="detail" class="d-flex align-center ga-2">
            <span class="status-chip" :style="{ color: statusCssColor(detail.status) }">
              <span
                class="task-dot"
                :style="{ background: statusCssColor(detail.status) }"
              />
              {{ statusLabel(detail.status) }}
            </span>
            <v-btn
              v-if="detail.active"
              variant="tonal"
              color="error"
              size="small"
              prepend-icon="$stop"
              @click="cancelSelected"
            >
              {{ t("detail.cancel", "取消任务") }}
            </v-btn>
          </div>
        </div>
        <v-divider />

        <v-card-text class="pa-4">
          <div v-if="!detail" class="detail-empty">
            <span class="detail-empty__icon">
              <v-icon icon="$image" size="22" />
            </span>
            <div class="detail-empty__title">
              {{ t("detail.emptyTitle", "选择一个任务查看详情") }}
            </div>
            <div class="section-caption text-center">
              {{ t("detail.emptyDescription", "任务结果、错误信息和下载入口会显示在这里。") }}
            </div>
          </div>

          <div v-else>
            <div class="detail-head">
              <span class="mono detail-id">{{ detail.task_id }}</span>
              <span class="section-caption">{{ formatDateTime(detail.created_at) }}</span>
            </div>

            <div class="prompt-block">
              <div class="prompt-label">{{ t("form.prompt", "提示词") }}</div>
              <div class="prompt-text">{{ promptExpanded ? promptText : promptPreview }}</div>
              <v-btn
                v-if="hasMorePrompt"
                variant="text"
                size="x-small"
                color="primary"
                class="px-0 mt-1"
                @click="promptExpanded = !promptExpanded"
              >
                {{
                  promptExpanded
                    ? t("detail.collapsePrompt", "收起提示词")
                    : t("detail.expandPrompt", "展开完整提示词")
                }}
              </v-btn>
            </div>

            <v-progress-linear
              v-if="detail.active"
              :model-value="detail.progress_percent || 0"
              color="primary"
              height="5"
              rounded
              class="my-4"
            />

            <div class="stat-grid">
              <div class="stat">
                <div class="stat-label">{{ t("detail.model", "模型") }}</div>
                <div class="stat-value">{{ detail.model || "--" }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">{{ t("detail.source", "任务来源") }}</div>
                <div class="stat-value">{{ detail.source || "--" }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">{{ t("detail.duration", "耗时") }}</div>
                <div class="stat-value">{{ formatDuration(detail.duration_seconds) }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">{{ t("form.aspectRatio", "宽高比") }}</div>
                <div class="stat-value">{{ detail.aspect_ratio || "--" }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">{{ t("form.resolution", "分辨率") }}</div>
                <div class="stat-value">{{ detail.resolution || "--" }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">{{ t("detail.requested", "请求数量") }}</div>
                <div class="stat-value">
                  {{ detail.result_count || 0 }} / {{ detail.requested_count || 0 }}
                </div>
              </div>
              <div class="stat">
                <div class="stat-label">{{ t("detail.references", "参考图") }}</div>
                <div class="stat-value">{{ detail.reference_image_count || 0 }}</div>
              </div>
              <div class="stat stat--wide">
                <div class="stat-label">{{ t("detail.userOrigin", "用户 UMO") }}</div>
                <div class="stat-value">{{ detail.unified_msg_origin || "--" }}</div>
              </div>
              <div v-if="detail.template_summary" class="stat stat--wide">
                <div class="stat-label">{{ t("detail.template", "模板") }}</div>
                <div class="stat-value">{{ detail.template_summary }}</div>
              </div>
            </div>

            <v-alert
              v-if="detail.error"
              type="error"
              variant="tonal"
              density="compact"
              class="mt-4"
              :text="detail.error"
            />

            <div class="section-head mt-5 mb-2">
              <span class="card-title-text">{{ t("detail.results", "生成结果") }}</span>
              <span class="section-caption">
                {{ detail.result_images?.length || 0 }}
              </span>
            </div>
            <div v-if="!detail.result_images?.length" class="section-caption">
              {{ t("detail.noResults", "暂无生成结果") }}
            </div>
            <div v-else class="result-grid">
              <div
                v-for="image in detail.result_images"
                :key="image.index"
                class="result-cell"
              >
                <div
                  class="thumb"
                  :class="{ 'thumb--empty': !image.available }"
                  @click="image.available && openImage(image)"
                >
                  <img
                    v-if="previewSrc(detail.task_id, image.index)"
                    :src="previewSrc(detail.task_id, image.index)"
                    alt=""
                  />
                  <span v-else class="section-caption text-center pa-1">
                    {{ t("detail.previewUnavailable", "暂无预览") }}
                  </span>
                </div>
                <div class="result-cell__bar">
                  <span class="section-caption">#{{ image.index }}</span>
                  <v-btn
                    icon="$download"
                    size="x-small"
                    variant="text"
                    :disabled="!image.available"
                    @click="downloadImage(image)"
                  />
                </div>
              </div>
            </div>

            <div class="section-head mt-5 mb-2">
              <span class="card-title-text">{{ t("detail.items", "子任务") }}</span>
            </div>
            <div v-if="!detail.items?.length" class="section-caption">
              {{ t("detail.noItems", "暂无子任务详情") }}
            </div>
            <v-table v-else density="compact">
              <thead>
                <tr>
                  <th>#</th>
                  <th>{{ t("detail.items", "子任务") }}</th>
                  <th>{{ t("detail.results", "生成结果") }}</th>
                  <th>{{ t("detail.error", "错误") }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in detail.items" :key="item.index">
                  <td>{{ item.index }}</td>
                  <td>
                    <span
                      class="status-chip"
                      :style="{ color: statusCssColor(item.status) }"
                    >
                      <span
                        class="task-dot"
                        :style="{ background: statusCssColor(item.status) }"
                      />
                      {{ statusLabel(item.status) }}
                    </span>
                  </td>
                  <td>{{ item.result_count }}</td>
                  <td class="section-caption">{{ item.error || "--" }}</td>
                </tr>
              </tbody>
            </v-table>
          </div>
        </v-card-text>
      </v-card>
    </v-col>
  </v-row>

  <ImageLightbox
    v-model="lightbox.open"
    :src="lightbox.src"
    :title="lightbox.title"
    :download-endpoint="lightbox.downloadEndpoint"
    :download-text="t('detail.download', '下载图片')"
    :close-text="t('lightbox.close', '关闭')"
    :empty-text="t('detail.previewUnavailable', '暂无预览')"
    @download="downloadLightbox"
  />
</template>

<script setup>
import {
  computed,
  inject,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
  watch,
} from "vue";
import { apiGet, apiPost, downloadFile } from "../api";
import { t } from "../i18n";
import {
  formatDateTime,
  formatDuration,
  statusCssColor,
  statusLabel,
} from "../utils";
import { previewKey, usePreview } from "../usePreview";
import ImageLightbox from "../components/ImageLightbox.vue";

const props = defineProps({
  refreshKey: { type: Number, default: 0 },
});

const notify = inject("notify", () => {});
const { cache, loadPreview } = usePreview();

const filters = reactive({
  keyword: "",
  status: "all",
  source: "",
  model: "",
  days: 0,
  limit: 30,
});

const tasks = ref([]);
const total = ref(0);
const offset = ref(0);
const loading = ref(false);
const availableModels = ref([]);
const availableSources = ref([]);
const selectedId = ref("");
const detail = ref(null);
const promptExpanded = ref(false);
const lightbox = reactive({
  open: false,
  src: "",
  title: "",
  downloadEndpoint: "",
  filename: "",
});

let pollTimer = null;
let searchTimer = null;
let pendingReload = null;
let preserveSelection = false;

const statusOptions = computed(() => [
  { title: t("filter.all", "全部"), value: "all" },
  { title: t("filter.active", "进行中"), value: "active" },
  { title: t("status.queued", "排队中"), value: "queued" },
  { title: t("status.running", "生成中"), value: "running" },
  { title: t("status.succeeded", "已完成"), value: "succeeded" },
  { title: t("status.failed", "失败"), value: "failed" },
  { title: t("status.cancelled", "已取消"), value: "cancelled" },
]);

const sourceOptions = computed(() => [
  { title: t("tasks.sourceAll", "全部来源"), value: "" },
  ...availableSources.value.map((source) => ({ title: source, value: source })),
]);

const modelOptions = computed(() => [
  { title: t("tasks.modelAll", "全部模型"), value: "" },
  ...availableModels.value.map((model) => ({ title: model, value: model })),
]);

const dayOptions = computed(() => [
  { title: t("gallery.daysAll", "全部时间"), value: 0 },
  { title: t("gallery.days1", "近 1 天"), value: 1 },
  { title: t("gallery.days7", "近 7 天"), value: 7 },
  { title: t("gallery.days30", "近 30 天"), value: 30 },
]);

const pages = computed(() =>
  Math.max(1, Math.ceil(total.value / Math.max(1, filters.limit))),
);

const countText = computed(() => ({
  total: total.value,
  page: Math.floor(offset.value / Math.max(1, filters.limit)) + 1,
  pages: pages.value,
}));

const promptText = computed(() => detail.value?.prompt || "");
const promptPreview = computed(
  () =>
    detail.value?.prompt_summary ||
    (promptText.value.length > 160
      ? `${promptText.value.slice(0, 160)}…`
      : promptText.value),
);
const hasMorePrompt = computed(
  () => promptText.value.length > promptPreview.value.length,
);

const previewSrc = (taskId, index) =>
  cache[previewKey(taskId, index)]?.data_url || "";

function debouncedReload() {
  if (searchTimer) {
    window.clearTimeout(searchTimer);
  }
  searchTimer = window.setTimeout(() => reload(true), 300);
}

async function reload(resetOffset, pageDelta = 0) {
  if (loading.value) {
    pendingReload = { resetOffset, pageDelta };
    return;
  }
  if (resetOffset) {
    offset.value = 0;
  } else if (pageDelta) {
    offset.value = Math.max(0, offset.value + pageDelta * filters.limit);
  }
  loading.value = true;
  try {
    const result = await apiGet("page/tasks", {
      status: filters.status,
      source: filters.source,
      model: filters.model,
      days: filters.days,
      limit: filters.limit,
      offset: offset.value,
      keyword: filters.keyword.trim(),
    });
    tasks.value = result?.tasks || [];
    total.value = result?.total || 0;
    availableModels.value = result?.models || [];
    availableSources.value = result?.sources || [];
    if (
      !preserveSelection &&
      !tasks.value.some((task) => task.task_id === selectedId.value)
    ) {
      selectedId.value = tasks.value[0]?.task_id || "";
    }
    preserveSelection = false;
    if (selectedId.value) {
      await selectTask(selectedId.value, { silent: true });
    } else {
      detail.value = null;
    }
    syncPolling();
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    loading.value = false;
    if (pendingReload) {
      const next = pendingReload;
      pendingReload = null;
      reload(next.resetOffset, next.pageDelta);
    }
  }
}

async function selectTask(taskId, { silent = false } = {}) {
  if (!taskId) {
    return;
  }
  if (taskId !== selectedId.value) {
    promptExpanded.value = false;
  }
  selectedId.value = taskId;
  try {
    const result = await apiGet(`page/tasks/${encodeURIComponent(taskId)}`);
    if (selectedId.value !== taskId) {
      return;
    }
    detail.value = result?.task || null;
    for (const image of detail.value?.result_images || []) {
      loadPreview(detail.value.task_id, image.index);
    }
    syncPolling();
  } catch (error) {
    if (!silent) {
      notify(error?.message || t("error.generic", "操作失败"), "error");
    }
  }
}

function syncPolling() {
  const hasActive = detail.value?.active || tasks.value.some((task) => task.active);
  if (hasActive && !pollTimer) {
    pollTimer = window.setInterval(pollActive, 4000);
  }
  if (!hasActive && pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollActive() {
  const result = await apiGet("page/tasks", {
    status: filters.status,
    source: filters.source,
    model: filters.model,
    days: filters.days,
    limit: filters.limit,
    offset: offset.value,
    keyword: filters.keyword.trim(),
  }).catch(() => null);
  if (result) {
    tasks.value = result.tasks || [];
    total.value = result.total || 0;
  }
  if (selectedId.value) {
    const detailResult = await apiGet(
      `page/tasks/${encodeURIComponent(selectedId.value)}`,
    ).catch(() => null);
    if (detailResult?.task) {
      detail.value = detailResult.task;
    }
  }
  syncPolling();
}

async function cancelSelected() {
  if (!detail.value) {
    return;
  }
  try {
    await apiPost(`page/tasks/${encodeURIComponent(detail.value.task_id)}/cancel`, {});
    notify(t("message.cancelled", "已请求取消任务"));
    await selectTask(detail.value.task_id, { silent: true });
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  }
}

async function openImage(image) {
  lightbox.title = `${detail.value?.task_id || ""} #${image.index}`;
  lightbox.downloadEndpoint = image.download_endpoint;
  lightbox.filename = image.filename || "image.png";
  lightbox.src = previewSrc(detail.value?.task_id, image.index);
  lightbox.open = true;
  const preview = await loadPreview(detail.value.task_id, image.index);
  if (preview?.data_url) {
    lightbox.src = preview.data_url;
  }
}

function downloadImage(image) {
  downloadFile(image.download_endpoint, image.filename || "image.png");
}

function downloadLightbox() {
  if (lightbox.downloadEndpoint) {
    downloadFile(lightbox.downloadEndpoint, lightbox.filename);
  }
}

function handleOpenTask(event) {
  const taskId = String(event?.detail || "");
  if (taskId) {
    preserveSelection = true;
    selectTask(taskId);
  }
}

watch(
  () => props.refreshKey,
  () => reload(true),
);

onMounted(() => {
  window.addEventListener("image-generation:open-task", handleOpenTask);
  reload(true);
});

onBeforeUnmount(() => {
  window.removeEventListener("image-generation:open-task", handleOpenTask);
  if (pollTimer) {
    window.clearInterval(pollTimer);
  }
  if (searchTimer) {
    window.clearTimeout(searchTimer);
  }
});
</script>

<style scoped>
.filter-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.filter-keyword {
  grid-column: 1 / -1;
}

.filter-count {
  grid-column: 1 / -1;
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.55);
}

.task-list {
  max-height: 54vh;
  overflow-y: auto;
}

.task-rows {
  display: flex;
  flex-direction: column;
  padding: 6px 10px 12px;
  gap: 8px;
}

.task-row {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 12px 14px 12px 18px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.09);
  border-radius: 10px;
  background: rgba(var(--v-theme-on-surface), 0.015);
  text-align: left;
  color: inherit;
  cursor: pointer;
  transition:
    background-color 0.15s ease,
    border-color 0.15s ease;
}

.task-row::before {
  content: "";
  position: absolute;
  left: 0;
  top: 10px;
  bottom: 10px;
  width: 3px;
  border-radius: 0 3px 3px 0;
  background: var(--status-color, rgba(var(--v-theme-on-surface), 0.3));
}

.task-row:hover {
  background: rgba(var(--v-theme-on-surface), 0.05);
  border-color: rgba(var(--v-theme-on-surface), 0.16);
}

.task-row--active {
  background: rgba(var(--v-theme-primary), 0.08);
  border-color: rgba(var(--v-theme-primary), 0.38);
}

.task-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  flex: none;
}

.task-row__main {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  flex: 1;
}

.task-row__title {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  font-size: 0.875rem;
  line-height: 1.55;
}

.task-row__meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.5);
  overflow: hidden;
  white-space: nowrap;
}

.dot-sep {
  opacity: 0.6;
}

.task-row__side {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  flex: none;
}

.task-row__count {
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.65);
}

.task-row__status {
  font-size: 0.75rem;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  font-weight: 600;
}

.list-empty {
  padding: 48px 16px;
  text-align: center;
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.45);
}

.detail-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 72px 16px;
}

.detail-empty__icon {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: rgba(var(--v-theme-primary), 0.12);
  color: rgb(var(--v-theme-primary));
  margin-bottom: 4px;
}

.detail-empty__title {
  font-size: 0.95rem;
  font-weight: 600;
}

.detail-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.detail-id {
  color: rgba(var(--v-theme-on-surface), 0.6);
  font-size: 11px;
}

.prompt-block {
  padding: 12px 14px;
  border-radius: 8px;
  background: rgba(var(--v-theme-on-surface), 0.04);
}

.prompt-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: rgba(var(--v-theme-on-surface), 0.55);
  margin-bottom: 4px;
}

.prompt-text {
  font-size: 0.875rem;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px 20px;
  margin-top: 16px;
}

.stat--wide {
  grid-column: span 3;
}

.stat-label {
  font-size: 0.72rem;
  color: rgba(var(--v-theme-on-surface), 0.5);
  margin-bottom: 2px;
}

.stat-value {
  font-size: 0.86rem;
  line-height: 1.5;
  word-break: break-word;
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
  gap: 12px;
}

.result-cell__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2px 2px 0;
}

@media (max-width: 600px) {
  .stat-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .stat--wide {
    grid-column: span 2;
  }
}
</style>
