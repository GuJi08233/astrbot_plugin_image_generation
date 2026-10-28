<template>
  <v-row dense>
    <v-col cols="12" lg="7">
      <v-card variant="outlined" class="rounded-lg h-100">
        <div class="card-title-row">
          <div>
            <div class="section-caption">{{ t("form.eyebrow", "Create") }}</div>
            <div class="card-title-text mt-1">{{ t("form.title", "创建生图任务") }}</div>
          </div>
          <v-btn
            color="primary"
            variant="flat"
            prepend-icon="$send"
            :loading="submitting"
            :disabled="!canSubmit"
            @click="submit"
          >
            {{ submitting ? t("submitting", "提交中...") : t("submit", "开始生成") }}
          </v-btn>
        </div>
        <v-divider />

        <v-card-text class="pa-4 d-flex flex-column ga-4">
          <v-textarea
            v-model="form.prompt"
            :label="t('form.prompt', '提示词')"
            :placeholder="t('form.promptPlaceholder', '描述你想生成的画面，也可以只选择预设或人设。')"
            rows="6"
            auto-grow
            variant="outlined"
            hide-details
          />

          <div v-if="supportsNegativePrompt">
            <v-checkbox
              v-model="negative.useDefault"
              :label="t('form.useDefaultNegativePrompt', '使用供应商默认反向提示词')"
              density="compact"
              hide-details
            />
            <v-textarea
              v-model="negative.text"
              :label="t('form.negativePrompt', '反向提示词')"
              :placeholder="t('form.negativePromptPlaceholder', '填写本次希望避免的内容。取消使用默认值并留空，可清空本次反向提示词。')"
              :hint="negativePromptHint"
              :disabled="negative.useDefault"
              rows="2"
              auto-grow
              variant="outlined"
              persistent-hint
            />
          </div>

          <v-row dense>
            <v-col cols="12" md="6">
              <v-select
                v-model="form.model"
                :items="modelOptions"
                :label="t('form.model', '模型')"
                :hint="modelHint"
                persistent-hint
                variant="outlined"
                density="comfortable"
                prepend-inner-icon="$image"
              />
            </v-col>
            <v-col cols="6" md="2">
              <v-text-field
                v-model.number="form.imageCount"
                type="number"
                :min="1"
                :max="maxImageCount"
                :label="t('form.imageCount', '生成数量')"
                variant="outlined"
                density="comfortable"
                hide-details
              />
            </v-col>
            <template v-if="isNovelai">
              <v-col v-for="key in ['width', 'height']" :key="key" cols="6" md="2">
                <v-text-field
                  v-model.number="form[key]"
                  type="number"
                  min="64"
                  max="2048"
                  step="64"
                  :label="key === 'width' ? t('form.imageWidth', '图片宽度（像素）') : t('form.imageHeight', '图片高度（像素）')"
                  variant="outlined"
                  density="comfortable"
                  hide-details
                />
              </v-col>
              <v-col cols="12" class="section-caption pt-0">
                {{ t("form.imageSizeHint", "宽高须为 64–2048 内的 64 倍数，总面积不超过 3,145,728 像素。自定义尺寸可能收费，仍受供应商 Gems 报价上限限制。") }}
              </v-col>
            </template>
            <template v-else>
              <v-col cols="6" md="2">
                <v-select
                  v-model="form.aspectRatio"
                  :items="aspectRatioOptions"
                  :disabled="!supportsAspectRatio"
                  :label="t('form.aspectRatio', '宽高比')"
                  variant="outlined"
                  density="comfortable"
                  hide-details
                />
              </v-col>
              <v-col cols="6" md="2">
                <v-select
                  v-model="form.resolution"
                  :items="resolutionOptions"
                  :disabled="!supportsResolution"
                  :label="t('form.resolution', '分辨率')"
                  variant="outlined"
                  density="comfortable"
                  hide-details
                />
              </v-col>
            </template>
          </v-row>

          <v-row dense>
            <v-col cols="12" md="6">
              <div class="field-label">
                <span>{{ t("form.presets", "预设") }}</span>
                <span class="section-caption">{{ t("form.optional", "可选") }}</span>
              </div>
              <div v-if="!presets.length" class="section-caption">
                {{ t("form.noPresets", "暂无预设") }}
              </div>
              <v-chip-group v-else v-model="selectedPresets" multiple column>
                <v-chip
                  v-for="preset in presets"
                  :key="preset.name"
                  :value="preset.name"
                  filter
                  size="small"
                  variant="outlined"
                  :title="preset.summary"
                >
                  {{ preset.name }}
                </v-chip>
              </v-chip-group>
            </v-col>

            <v-col cols="12" md="6">
              <div class="field-label">
                <span>{{ t("form.personas", "人设") }}</span>
                <span class="section-caption">{{ t("form.optional", "可选") }}</span>
              </div>
              <div v-if="!personas.length" class="section-caption">
                {{ t("form.noPersonas", "暂无人设") }}
              </div>
              <v-chip-group v-else v-model="selectedPersonas" multiple column>
                <v-chip
                  v-for="persona in personas"
                  :key="persona.name"
                  :value="persona.name"
                  filter
                  size="small"
                  variant="outlined"
                  :prepend-icon="persona.has_image ? '$image' : undefined"
                  :title="persona.summary"
                >
                  {{ persona.name }}
                </v-chip>
              </v-chip-group>
            </v-col>
          </v-row>

          <div>
            <v-file-input
              v-model="pendingFiles"
              :label="isNovelai ? t('novelai.sourceImage', '图生图 / 局部重绘源图（最多 1 张）') : t('form.referenceImages', '参考图')"
              :hint="referenceHint"
              :disabled="!supportsImageToImage || uploading || (isNovelai && novelai.mode === 'generate')"
              :loading="uploading"
              accept="image/*"
              :multiple="!isNovelai"
              chips
              show-size
              variant="outlined"
              density="comfortable"
              persistent-hint
              prepend-icon=""
              prepend-inner-icon="$image"
              @update:model-value="handleFiles"
            />
            <div v-if="uploads.length" class="reference-grid mt-3">
              <div
                v-for="upload in uploads"
                :key="upload.token"
                class="reference-item"
              >
                <div
                  class="thumb reference-thumb"
                  :title="t('form.openReference', '点击查看大图')"
                  @click="openReference(upload)"
                >
                  <img
                    v-if="upload.preview_url"
                    :src="upload.preview_url"
                    :alt="upload.filename"
                  />
                  <v-icon v-else icon="$image" size="18" />
                  <v-btn
                    class="reference-remove"
                    icon="$close"
                    size="x-small"
                    color="surface"
                    variant="flat"
                    :title="t('form.removeReference', '移除参考图')"
                    @click.stop="removeUpload(upload.token)"
                  />
                </div>
                <div class="reference-name" :title="upload.filename">
                  {{ upload.filename }}
                </div>
                <div class="section-caption">{{ formatBytes(upload.size) }}</div>
              </div>
            </div>
            <div v-if="uploads.length" class="reference-count section-caption mt-2">
              {{
                t("form.referenceCount", "已选择 {count} 张参考图", {
                  count: uploads.length,
                })
              }}
            </div>
          </div>

          <NovelaiOptions
            v-if="isNovelai"
            v-model="novelai"
            :model="selectedModelCapabilities?.model || ''"
          />
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" lg="5">
      <v-card variant="outlined" class="rounded-lg h-100">
        <div class="card-title-row">
          <span class="card-title-text">{{ t("form.resultTitle", "本次生成") }}</span>
          <v-btn
            v-if="results.length"
            variant="text"
            size="small"
            @click="clearResults"
          >
            {{ t("form.clearResult", "关闭结果") }}
          </v-btn>
        </div>
        <v-divider />

        <v-card-text class="pa-4">
          <div v-if="!results.length" class="result-empty">
            <span class="result-empty__icon">
              <v-icon icon="$images" size="22" />
            </span>
            <div class="result-empty__title">
              {{ t("form.resultEmptyTitle", "提交后在此查看结果") }}
            </div>
            <div class="section-caption text-center">
              {{ t("form.resultEmptyDescription", "生成进度、错误信息和图片结果会显示在这里，无需离开当前页面。") }}
            </div>
          </div>

          <div
            v-for="task in results"
            :key="task.task_id"
            class="result-task"
          >
            <div class="d-flex align-center justify-space-between mb-2">
              <div class="d-flex align-center ga-2">
                <v-chip :color="statusColor(task.status)" size="x-small" variant="tonal">
                  {{ statusLabel(task.status) }}
                </v-chip>
                <span class="section-caption">{{ task.model || "--" }}</span>
              </div>
              <div class="d-flex align-center ga-1">
                <v-btn
                  v-if="task.active"
                  icon="$stop"
                  size="x-small"
                  variant="text"
                  color="error"
                  @click="cancelTask(task.task_id)"
                />
                <v-btn
                  icon="$openInNew"
                  size="x-small"
                  variant="text"
                  :title="t('form.openTask', '在任务页查看')"
                  @click="emit('openTask', task.task_id)"
                />
              </div>
            </div>

            <div class="prompt-clamp section-caption mb-2">
              {{ task.prompt_summary || task.prompt || "--" }}
            </div>

            <v-progress-linear
              v-if="task.active"
              :model-value="task.progress_percent || 0"
              color="primary"
              height="5"
              rounded
              class="mb-2"
            />
            <div class="mono section-caption mb-2">
              {{ task.current_index || 0 }} / {{ task.requested_count || 1 }}
              · {{ formatDuration(task.duration_seconds) }}
            </div>

            <v-alert
              v-if="task.error"
              type="error"
              variant="tonal"
              density="compact"
              class="mb-2"
              :text="task.error"
            />

            <AuditResults :entries="task.audit_results || []" class="mb-2" />

            <div v-if="task.result_images?.length" class="result-grid">
              <div
                v-for="image in task.result_images"
                :key="image.index"
                class="thumb result-thumb"
                @click="openImage(task, image)"
              >
                <img
                  v-if="previewSrc(task.task_id, image.index)"
                  :src="previewSrc(task.task_id, image.index)"
                  alt=""
                />
                <v-progress-circular v-else indeterminate size="16" />
              </div>
            </div>
          </div>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12">
      <AuditTestCard />
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
import { apiGet, apiPost, downloadFile, uploadFile } from "../api";
import { t } from "../i18n";
import {
  ASPECT_RATIOS,
  RESOLUTIONS,
  formatBytes,
  formatDuration,
  statusColor,
  statusLabel,
} from "../utils";
import { previewKey, usePreview } from "../usePreview";
import { buildNovelaiOptions, isValidPixelSize } from "../novelai";
import AuditResults from "../components/AuditResults.vue";
import AuditTestCard from "../components/AuditTestCard.vue";
import ImageLightbox from "../components/ImageLightbox.vue";
import NovelaiOptions from "../components/NovelaiOptions.vue";

const props = defineProps({
  pluginState: { type: Object, default: null },
  refreshKey: { type: Number, default: 0 },
});

const emit = defineEmits(["openTask"]);
const notify = inject("notify", () => {});
const { cache, loadPreview } = usePreview();

const form = reactive({
  prompt: "",
  model: "",
  imageCount: 1,
  aspectRatio: "不指定",
  resolution: "不指定",
  width: 832,
  height: 1216,
});
// Unchecked "use default" sends the text (even empty) as a per-task override.
const negative = reactive({ useDefault: true, text: "" });
// NovelAI advanced inputs survive submissions so a setup can be reused.
const novelai = ref({
  mode: "auto",
  parametersText: "",
  mask: null,
  references: [],
  vibeText: "",
  vibeImport: null,
  vibeVariant: "",
});

const selectedPresets = ref([]);
const selectedPersonas = ref([]);
const pendingFiles = ref([]);
const uploads = ref([]);
const uploading = ref(false);
const submitting = ref(false);
const results = ref([]);
const lightbox = reactive({
  open: false,
  src: "",
  title: "",
  downloadEndpoint: "",
  filename: "",
});

let pollTimer = null;
let formInitialized = false;

const presets = computed(() => props.pluginState?.presets || []);
const personas = computed(() => props.pluginState?.personas || []);
const maxImageCount = computed(() => props.pluginState?.max_image_count || 8);
// Capability flags follow the model picked in the form; the active model only
// acts as the fallback while the model list is unavailable.
const selectedModelCapabilities = computed(
  () =>
    (props.pluginState?.model_capabilities || {})[form.model] ||
    props.pluginState,
);
const supportsAspectRatio = computed(
  () => selectedModelCapabilities.value?.supports_aspect_ratio !== false,
);
const supportsResolution = computed(
  () => selectedModelCapabilities.value?.supports_resolution !== false,
);
const supportsImageToImage = computed(
  () => selectedModelCapabilities.value?.supports_image_to_image !== false,
);
const supportsNegativePrompt = computed(
  () => selectedModelCapabilities.value?.supports_negative_prompt === true,
);
const isNovelai = computed(
  () => selectedModelCapabilities.value?.supports_novelai === true,
);
const negativePromptHint = computed(() => {
  const capabilities = selectedModelCapabilities.value || {};
  if (!["openai", "openai_chat"].includes(capabilities.adapter_type)) {
    return t("form.negativePromptHint", "与正向提示词用于同一次生图。仅覆盖本次反向词，不修改供应商默认配置。");
  }
  return capabilities.negative_prompt_mode === "parameter"
    ? t("form.negativePromptParameterHint", "本次反向词将作为独立 negative_prompt 字段随同一次请求发送；需要兼容服务明确支持该扩展。不会修改供应商默认值。")
    : t("form.negativePromptPromptHint", "本次反向词会与正向词合并为同一次请求中的语义约束，不使用独立负向通道。不会修改供应商默认值。");
});

const modelOptions = computed(() =>
  (props.pluginState?.available_models || []).map((model) => ({
    title: model,
    value: model,
  })),
);

const modelHint = computed(() =>
  modelOptions.value.length ? "" : t("form.noModels", "使用当前配置模型"),
);

const aspectRatioOptions = computed(() =>
  ASPECT_RATIOS.map((value) => ({
    title: value === "不指定" ? t("form.unspecified", "不指定") : value,
    value,
  })),
);

const resolutionOptions = computed(() =>
  RESOLUTIONS.map((value) => ({
    title: value === "不指定" ? t("form.unspecified", "不指定") : value,
    value,
  })),
);

const referenceHint = computed(() => {
  if (isNovelai.value) {
    return t("novelai.sourceHint", "自动模式下有源图则图生图，有蒙版则局部重绘。文生图模式忽略源图；角色参考与 Vibe 在高级设置中单独填写。");
  }
  return supportsImageToImage.value
    ? t("form.referenceHint", "支持 jpg/png/webp/gif/heic/heif，按配置限制大小。")
    : t("form.referenceUnsupported", "当前适配器未声明图生图能力。");
});

const canSubmit = computed(
  () =>
    !submitting.value &&
    !uploading.value &&
    (form.prompt.trim().length > 0 ||
      selectedPresets.value.length > 0 ||
      selectedPersonas.value.length > 0 ||
      // NovelAI characters can carry the whole prompt.
      (isNovelai.value && novelai.value.parametersText.trim().length > 0)),
);

const previewSrc = (taskId, index) =>
  cache[previewKey(taskId, index)]?.data_url || "";

async function handleFiles(files) {
  const list = Array.isArray(files) ? files : files ? [files] : [];
  if (!list.length) {
    return;
  }
  if (isNovelai.value && list.length + uploads.value.length > 1) {
    notify(t("novelai.oneSource", "NovelAI 图生图或局部重绘最多使用一张源图，蒙版也必须单独上传一张。"), "error");
    pendingFiles.value = [];
    return;
  }
  uploading.value = true;
  try {
    for (const file of list) {
      const upload = await uploadFile("page/reference/upload", file);
      // Keep a local object URL so the uploaded image can be previewed before
      // the task is submitted; released again when the entry goes away.
      uploads.value.push({ ...upload, preview_url: URL.createObjectURL(file) });
    }
    notify(t("message.uploaded", "参考图已上传"));
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    uploading.value = false;
    pendingFiles.value = [];
  }
}

function releaseUploadPreviews(items) {
  for (const item of items) {
    if (item?.preview_url) {
      URL.revokeObjectURL(item.preview_url);
    }
  }
}

function removeUpload(token) {
  releaseUploadPreviews(
    uploads.value.filter((item) => item.token === token),
  );
  uploads.value = uploads.value.filter((item) => item.token !== token);
}

async function submit() {
  if (!canSubmit.value) {
    return;
  }
  submitting.value = true;
  try {
    const payload = {
      prompt: form.prompt.trim(),
      model: form.model || "",
      image_count: form.imageCount || 1,
      aspect_ratio: form.aspectRatio,
      resolution: form.resolution,
      presets: selectedPresets.value,
      personas: selectedPersonas.value,
      reference_tokens: uploads.value.map((item) => item.token),
    };
    if (supportsNegativePrompt.value && !negative.useDefault) {
      payload.negative_prompt = negative.text.trim();
    }
    if (isNovelai.value) {
      // NovelAI uses explicit pixels instead of ratio/resolution presets.
      if (!isValidPixelSize(form.width, form.height)) {
        throw new Error(
          t("form.invalidImageSize", "请填写 64–2048 内的 64 倍数宽高，且总面积不超过 3,145,728 像素。"),
        );
      }
      payload.width = form.width;
      payload.height = form.height;
      delete payload.aspect_ratio;
      delete payload.resolution;
      const options = buildNovelaiOptions(
        { ...novelai.value, model: selectedModelCapabilities.value?.model || "" },
        payload,
        t,
      );
      if (options) {
        payload.novelai = options;
      }
    }
    const result = await apiPost("page/generate", payload);
    const task =
      result.task ||
      (result.task_id ? { task_id: result.task_id, active: true } : null);
    if (task) {
      results.value = [
        task,
        ...results.value.filter((item) => item.task_id !== task.task_id),
      ];
      loadTaskPreviews(task);
      startPolling();
    }
    notify(t("message.taskSubmitted", "任务已提交"));
    form.prompt = "";
    releaseUploadPreviews(uploads.value);
    uploads.value = [];
    selectedPresets.value = [];
    selectedPersonas.value = [];
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    submitting.value = false;
  }
}

function loadTaskPreviews(task) {
  for (const image of task?.result_images || []) {
    loadPreview(task.task_id, image.index);
  }
}

async function refreshResults() {
  const active = results.value.filter((task) => task.active);
  if (!active.length) {
    stopPolling();
    return;
  }
  await Promise.all(
    active.map(async (task) => {
      try {
        const result = await apiGet(
          `page/tasks/${encodeURIComponent(task.task_id)}`,
        );
        const index = results.value.findIndex(
          (item) => item.task_id === task.task_id,
        );
        if (index >= 0 && result?.task) {
          results.value[index] = result.task;
          loadTaskPreviews(result.task);
        }
      } catch {
        // Keep the previous snapshot on transient failures.
      }
    }),
  );
  if (!results.value.some((task) => task.active)) {
    stopPolling();
  }
}

function startPolling() {
  if (pollTimer) {
    return;
  }
  pollTimer = window.setInterval(refreshResults, 3000);
}

function stopPolling() {
  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function cancelTask(taskId) {
  try {
    await apiPost(`page/tasks/${encodeURIComponent(taskId)}/cancel`, {});
    notify(t("message.cancelled", "已请求取消任务"));
    await refreshResults();
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  }
}

function clearResults() {
  results.value = [];
  stopPolling();
}

async function openImage(task, image) {
  lightbox.title = `${task.task_id} #${image.index}`;
  lightbox.downloadEndpoint = image.download_endpoint;
  lightbox.filename = image.filename || "image.png";
  lightbox.src = previewSrc(task.task_id, image.index);
  lightbox.open = true;
  const preview = await loadPreview(task.task_id, image.index);
  if (preview?.data_url) {
    lightbox.src = preview.data_url;
  }
}

function openReference(upload) {
  lightbox.title = upload.filename || t("form.referenceImages", "参考图");
  lightbox.downloadEndpoint = "";
  lightbox.filename = "";
  lightbox.src = upload.preview_url || "";
  lightbox.open = true;
}

function downloadLightbox() {
  if (lightbox.downloadEndpoint) {
    downloadFile(lightbox.downloadEndpoint, lightbox.filename);
  }
}

watch(
  () => props.refreshKey,
  async () => {
    if (results.value.some((task) => task.active)) {
      await refreshResults();
    }
  },
);

// Each NovelAI model starts from its provider's default pixel size.
watch(
  () => [form.model, isNovelai.value],
  () => {
    if (isNovelai.value) {
      form.width = selectedModelCapabilities.value?.novelai_width ?? 832;
      form.height = selectedModelCapabilities.value?.novelai_height ?? 1216;
    }
  },
);

watch(
  () => props.pluginState,
  (state) => {
    if (!state || formInitialized) {
      return;
    }
    formInitialized = true;
    form.model = state.current_model || "";
    form.imageCount = Math.min(
      Math.max(1, state.default_image_count || 1),
      state.max_image_count || 8,
    );
    form.aspectRatio = state.default_aspect_ratio || "不指定";
    form.resolution = state.default_resolution || "不指定";
  },
  { immediate: true },
);

onMounted(() => {
  if (results.value.some((task) => task.active)) {
    startPolling();
  }
});

onBeforeUnmount(() => {
  stopPolling();
  releaseUploadPreviews(uploads.value);
});
</script>

<style scoped>
.field-label {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.72);
}

.result-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 48px 16px;
}

.result-empty__icon {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: rgba(var(--v-theme-primary), 0.12);
  color: rgb(var(--v-theme-primary));
  margin-bottom: 4px;
}

.result-empty__title {
  font-size: 0.95rem;
  font-weight: 600;
}

.result-task + .result-task {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid rgba(var(--v-border-color), 0.24);
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.result-thumb {
  aspect-ratio: 1;
}

.reference-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(88px, 1fr));
  gap: 12px;
}

.reference-item {
  min-width: 0;
}

.reference-thumb {
  border: 1px solid rgba(var(--v-border-color), 0.3);
}

.reference-remove {
  position: absolute;
  top: 4px;
  right: 4px;
  opacity: 0.88;
}

.reference-name {
  margin-top: 6px;
  font-size: 0.75rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.reference-count {
  opacity: 0.72;
}
</style>
