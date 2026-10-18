<template>
  <v-card variant="outlined" class="rounded-lg">
    <div class="card-title-row">
      <div>
        <div class="section-caption">{{ t("gallery.eyebrow", "Gallery") }}</div>
        <div class="card-title-text mt-1">{{ t("gallery.title", "生成图库") }}</div>
      </div>
      <v-btn
        variant="tonal"
        size="small"
        prepend-icon="$refresh"
        :loading="loading"
        @click="reload"
      >
        {{ t("refresh", "刷新") }}
      </v-btn>
    </div>
    <v-divider />

    <div class="filter-bar pa-4 pb-2">
      <v-text-field
        v-model="filters.keyword"
        :placeholder="t('gallery.keywordPlaceholder', '任务 ID / 提示词 / 预设 / 模型')"
        density="compact"
        variant="outlined"
        hide-details
        clearable
        style="min-width: 200px; flex: 1 1 240px"
        @update:model-value="debouncedReload"
      />
      <v-select
        v-model="filters.model"
        :items="modelOptions"
        density="compact"
        variant="outlined"
        hide-details
        style="width: 170px"
        @update:model-value="reload"
      />
      <v-select
        v-model="filters.days"
        :items="dayOptions"
        density="compact"
        variant="outlined"
        hide-details
        style="width: 124px"
        @update:model-value="reload"
      />
      <v-select
        v-model="filters.limit"
        :items="[24, 48, 96, 150]"
        density="compact"
        variant="outlined"
        hide-details
        style="width: 96px"
        @update:model-value="reload"
      />
    </div>

    <div class="section-caption px-4 pb-2">
      {{ t("gallery.count", "共 {total} 张，可用 {available} 张", countText) }}
    </div>

    <v-divider />

    <v-card-text class="pa-4">
      <div v-if="!items.length" class="gallery-empty">
        <span class="gallery-empty__icon">
          <v-icon icon="$images" size="22" />
        </span>
        <div class="section-caption">{{ t("gallery.empty", "暂无生成图片") }}</div>
      </div>

      <div v-else class="gallery-grid">
        <v-card
          v-for="item in items"
          :key="item.id"
          variant="outlined"
          class="gallery-card rounded-lg"
        >
          <div
            class="thumb gallery-thumb"
            :class="{ 'thumb--empty': !item.available }"
            @click="item.available && openImage(item)"
          >
            <img
              v-if="previewSrc(item.task_id, item.image_index)"
              :src="previewSrc(item.task_id, item.image_index)"
              alt=""
            />
            <span v-else class="section-caption text-center pa-2">
              {{
                item.available
                  ? t("detail.previewFailed", "预览失败")
                  : t("gallery.unavailable", "文件已清理")
              }}
            </span>
          </div>

          <div class="gallery-body">
            <div class="gallery-meta">
              <span class="gallery-model">{{ item.model || "--" }}</span>
              <span class="section-caption">#{{ item.image_index }}</span>
            </div>
            <div class="gallery-prompt prompt-clamp">
              {{ item.prompt_summary || item.task_id }}
            </div>
            <div class="section-caption">{{ formatDateTime(item.generated_at) }}</div>
          </div>

          <div class="gallery-actions">
            <v-btn
              size="small"
              variant="text"
              prepend-icon="$list"
              @click="emit('openTask', item.task_id)"
            >
              {{ t("gallery.openTask", "查看任务") }}
            </v-btn>
            <v-spacer />
            <v-btn
              icon="$download"
              size="small"
              variant="text"
              :disabled="!item.available"
              @click="downloadItem(item)"
            />
          </div>
        </v-card>
      </div>
    </v-card-text>
  </v-card>

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
import { apiGet, downloadFile } from "../api";
import { t } from "../i18n";
import { formatDateTime } from "../utils";
import { previewKey, usePreview } from "../usePreview";
import ImageLightbox from "../components/ImageLightbox.vue";

const props = defineProps({
  refreshKey: { type: Number, default: 0 },
});

const emit = defineEmits(["openTask"]);
const notify = inject("notify", () => {});
const { cache, loadPreview } = usePreview();

const filters = reactive({
  keyword: "",
  model: "",
  days: 7,
  limit: 48,
});

const items = ref([]);
const total = ref(0);
const availableCount = ref(0);
const loading = ref(false);
const availableModels = ref([]);
const lightbox = reactive({
  open: false,
  src: "",
  title: "",
  downloadEndpoint: "",
  filename: "",
});

let searchTimer = null;

const modelOptions = computed(() => [
  { title: t("gallery.modelAll", "全部模型"), value: "" },
  ...availableModels.value.map((model) => ({ title: model, value: model })),
]);

const dayOptions = computed(() => [
  { title: t("gallery.daysAll", "全部时间"), value: 0 },
  { title: t("gallery.days1", "近 1 天"), value: 1 },
  { title: t("gallery.days7", "近 7 天"), value: 7 },
  { title: t("gallery.days30", "近 30 天"), value: 30 },
]);

const countText = computed(() => ({
  total: total.value,
  available: availableCount.value,
}));

const previewSrc = (taskId, index) =>
  cache[previewKey(taskId, index)]?.data_url || "";

function debouncedReload() {
  if (searchTimer) {
    window.clearTimeout(searchTimer);
  }
  searchTimer = window.setTimeout(reload, 300);
}

async function reload() {
  if (loading.value) {
    return;
  }
  loading.value = true;
  try {
    const result = await apiGet("page/gallery", {
      model: filters.model,
      days: filters.days,
      limit: filters.limit,
      keyword: filters.keyword.trim(),
    });
    items.value = result?.items || [];
    total.value = result?.total || 0;
    availableCount.value = result?.available_count || 0;
    availableModels.value = result?.models || [];
    items.value.forEach((item) => loadPreview(item.task_id, item.image_index));
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    loading.value = false;
  }
}

async function openImage(item) {
  lightbox.title = `${item.task_id} #${item.image_index}`;
  lightbox.downloadEndpoint = item.download_endpoint;
  lightbox.filename = item.filename || "image.png";
  lightbox.src = previewSrc(item.task_id, item.image_index);
  lightbox.open = true;
  const preview = await loadPreview(item.task_id, item.image_index);
  if (preview?.data_url) {
    lightbox.src = preview.data_url;
  }
}

function downloadItem(item) {
  downloadFile(item.download_endpoint, item.filename || "image.png");
}

function downloadLightbox() {
  if (lightbox.downloadEndpoint) {
    downloadFile(lightbox.downloadEndpoint, lightbox.filename);
  }
}

watch(
  () => props.refreshKey,
  () => reload(),
);

onMounted(reload);

onBeforeUnmount(() => {
  if (searchTimer) {
    window.clearTimeout(searchTimer);
  }
});
</script>

<style scoped>
.filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 16px;
}

.gallery-card {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.gallery-thumb {
  border-radius: 0;
  aspect-ratio: 1;
}

.gallery-body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
}

.gallery-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.gallery-model {
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.62);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gallery-prompt {
  font-size: 0.82rem;
  line-height: 1.5;
}

.gallery-actions {
  display: flex;
  align-items: center;
  padding: 0 4px 4px;
  margin-top: auto;
}

.gallery-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 64px 16px;
}

.gallery-empty__icon {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: rgba(var(--v-theme-primary), 0.12);
  color: rgb(var(--v-theme-primary));
}
</style>
