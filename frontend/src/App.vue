<template>
  <v-app :theme="themeName">
    <v-main class="page-main">
      <div class="page-shell">
        <header class="page-header">
          <div class="page-header__text">
            <h1 class="page-title">{{ t("title", "通用生图控制台") }}</h1>
          </div>

          <div class="page-header__actions">
            <v-select
              v-if="modelOptions.length"
              v-model="selectedModel"
              :items="modelOptions"
              :label="t('status.model', '当前模型')"
              :disabled="switchingModel"
              density="compact"
              variant="outlined"
              hide-details
              class="page-model-select"
              @update:model-value="switchModel"
            />
            <v-chip
              v-else-if="pluginState?.current_model"
              size="small"
              variant="tonal"
              color="primary"
              prepend-icon="$image"
            >
              {{ pluginState.current_model }}
            </v-chip>
            <v-chip
              v-if="pluginState && pluginState.initialized === false"
              size="small"
              variant="tonal"
              color="warning"
            >
              {{ t("status.notReady", "未初始化") }}
            </v-chip>
            <v-btn
              variant="tonal"
              prepend-icon="$refresh"
              :loading="loadingState"
              @click="reloadAll"
            >
              {{ t("refresh", "刷新") }}
            </v-btn>
          </div>
        </header>

        <v-tabs v-model="tab" color="primary" density="comfortable" class="page-tabs">
          <v-tab value="overview" :prepend-icon="'$dashboard'">
            {{ t("nav.overview", "概览") }}
          </v-tab>
          <v-tab value="generate" :prepend-icon="'$plus'">
            {{ t("nav.generate", "生成") }}
          </v-tab>
          <v-tab value="tasks" :prepend-icon="'$list'">
            {{ t("nav.tasks", "任务") }}
          </v-tab>
          <v-tab value="gallery" :prepend-icon="'$images'">
            {{ t("nav.gallery", "图库") }}
          </v-tab>
        </v-tabs>

        <v-alert
          v-if="loadError"
          type="error"
          variant="tonal"
          density="comfortable"
          class="mt-4"
          :text="loadError"
        />

        <v-window v-model="tab" class="page-window">
          <v-window-item value="overview">
            <OverviewView :refresh-key="refreshKey" />
          </v-window-item>
          <v-window-item value="generate">
            <GenerateView
              :plugin-state="pluginState"
              :refresh-key="refreshKey"
              @open-task="openTask"
            />
          </v-window-item>
          <v-window-item value="tasks">
            <TasksView :refresh-key="refreshKey" />
          </v-window-item>
          <v-window-item value="gallery">
            <GalleryView :refresh-key="refreshKey" @open-task="openTask" />
          </v-window-item>
        </v-window>
      </div>
    </v-main>

    <v-snackbar v-model="snackbar.show" :color="snackbar.color" timeout="2600">
      {{ snackbar.text }}
    </v-snackbar>
  </v-app>
</template>

<script setup>
import { computed, onMounted, provide, reactive, ref, watch } from "vue";
import { waitForBridge } from "./bridge";
import { apiGet, apiPost } from "./api";
import { isDark, syncContext, t } from "./i18n";
import { THEME_DARK, THEME_LIGHT } from "./theme";
import OverviewView from "./views/OverviewView.vue";
import GenerateView from "./views/GenerateView.vue";
import TasksView from "./views/TasksView.vue";
import GalleryView from "./views/GalleryView.vue";

const tab = ref("overview");
const pluginState = ref(null);
const loadingState = ref(false);
const loadError = ref("");
const refreshKey = ref(0);
const selectedModel = ref("");
const switchingModel = ref(false);

const snackbar = reactive({ show: false, text: "", color: "success" });

const themeName = computed(() => (isDark.value ? THEME_DARK : THEME_LIGHT));
const modelOptions = computed(() => pluginState.value?.available_models || []);

watch(
  () => pluginState.value?.current_model,
  (value) => {
    selectedModel.value = value || "";
  },
);

function notify(text, color = "success") {
  snackbar.text = text;
  snackbar.color = color;
  snackbar.show = true;
}

provide("notify", notify);

async function switchModel(model) {
  const previous = pluginState.value?.current_model || "";
  if (!model || model === previous) {
    return;
  }
  switchingModel.value = true;
  try {
    const result = await apiPost("page/model", { model });
    const current = result?.current_model || model;
    if (pluginState.value) {
      pluginState.value.current_model = current;
    }
    notify(
      t("status.modelSwitched", "模型已切换: {model}", { model: current }),
    );
    refreshKey.value += 1;
  } catch (error) {
    selectedModel.value = previous;
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    switchingModel.value = false;
  }
}

async function loadState() {
  loadingState.value = true;
  loadError.value = "";
  try {
    pluginState.value = await apiGet("page/state");
  } catch (error) {
    loadError.value = error?.message || t("error.generic", "操作失败");
  } finally {
    loadingState.value = false;
  }
}

async function reloadAll() {
  await loadState();
  refreshKey.value += 1;
}

function openTask(taskId) {
  tab.value = "tasks";
  refreshKey.value += 1;
  if (taskId) {
    window.setTimeout(() => {
      window.dispatchEvent(
        new CustomEvent("image-generation:open-task", { detail: taskId }),
      );
    }, 0);
  }
}

onMounted(async () => {
  const bridge = await waitForBridge();
  syncContext(bridge.getContext?.());
  if (typeof bridge.onContext === "function") {
    bridge.onContext(syncContext);
  }
  await loadState();
});
</script>

<style>
html,
body {
  margin: 0;
  padding: 0;
  background: rgb(var(--v-theme-background));
  color: rgb(var(--v-theme-on-background));
  font-family:
    "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, -apple-system,
    "Segoe UI", Roboto, sans-serif;
}

.v-application {
  background: rgb(var(--v-theme-background));
}

.page-main {
  background: rgb(var(--v-theme-background));
}

.page-shell {
  width: 100%;
  padding: 16px 16px 32px;
}

/* Vuetify 3.7 renders outlined cards with `currentColor`, which is far too
   heavy next to AstrBot's soft borders. */
.v-card--variant-outlined {
  border-color: rgba(var(--v-theme-on-surface), 0.12) !important;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.page-title {
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.25;
  margin: 0;
}

.page-model-select {
  flex: 1 1 240px;
  min-width: 200px;
  max-width: 320px;
}

.page-header__actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.page-tabs {
  border-bottom: 1px solid rgba(var(--v-border-color), 0.24);
}

.page-window {
  padding-top: 16px;
}

.section-title {
  font-size: 1.15rem;
  font-weight: 700;
  margin: 0;
}

.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.section-caption {
  font-size: 0.78rem;
  color: rgba(var(--v-theme-on-surface), 0.55);
}

.card-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 18px;
}

.card-title-text {
  font-size: 0.98rem;
  font-weight: 600;
}

.meta-label {
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.55);
  line-height: 1.4;
}

.meta-value {
  font-size: 0.875rem;
  line-height: 1.5;
  word-break: break-word;
}

.prompt-clamp {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.thumb {
  position: relative;
  display: grid;
  place-items: center;
  width: 100%;
  aspect-ratio: 1;
  overflow: hidden;
  border-radius: 8px;
  background: rgba(var(--v-theme-on-surface), 0.06);
  cursor: pointer;
}

.thumb img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumb--empty {
  cursor: default;
}

.lightbox-image {
  max-width: 100%;
  max-height: 72vh;
  object-fit: contain;
  border-radius: 8px;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 12px;
}
</style>
