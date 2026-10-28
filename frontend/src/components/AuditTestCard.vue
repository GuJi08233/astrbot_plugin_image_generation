<template>
  <v-card variant="outlined" class="rounded-lg">
    <div class="card-title-row">
      <div>
        <div class="section-caption">{{ t("audit.testEyebrow", "Audit") }}</div>
        <div class="card-title-text mt-1">{{ t("audit.testTitle", "审核测试") }}</div>
      </div>
      <v-btn
        color="primary"
        variant="tonal"
        prepend-icon="$audit"
        :loading="testing"
        :disabled="uploading"
        @click="runTest"
      >
        {{ testing ? t("audit.testRunning", "测试中...") : t("audit.testRun", "运行测试") }}
      </v-btn>
    </div>
    <v-divider />
    <v-card-text class="pa-4 d-flex flex-column ga-4">
      <div class="section-caption">
        {{ t("audit.testHint", "按当前安全审核配置直接测试，不生成图片、不产生任务记录。提示词和图片可以只填其中一项。") }}
      </div>
      <v-row dense>
        <v-col cols="12" md="6">
          <v-textarea
            v-model="prompt"
            :label="t('audit.testPrompt', '测试提示词')"
            :placeholder="t('audit.testPromptPlaceholder', '输入要测试的提示词，留空则跳过提示词审核。')"
            rows="3"
            auto-grow
            variant="outlined"
            hide-details
          />
        </v-col>
        <v-col cols="12" md="6">
          <v-file-input
            v-model="pendingFiles"
            :label="t('audit.testImage', '测试图片')"
            :hint="t('audit.testImageHint', '上传图片后按当前配置执行 Moderation / AI 图片审核。')"
            :loading="uploading"
            :disabled="uploading"
            accept="image/*"
            multiple
            variant="outlined"
            density="comfortable"
            persistent-hint
            prepend-icon=""
            prepend-inner-icon="$image"
            @update:model-value="handleFiles"
          />
          <div v-if="uploads.length" class="d-flex flex-wrap ga-2 mt-2">
            <v-chip
              v-for="upload in uploads"
              :key="upload.token"
              size="small"
              closable
              @click:close="uploads = uploads.filter((item) => item.token !== upload.token)"
            >
              {{ upload.filename }} · {{ formatBytes(upload.size) }}
            </v-chip>
          </div>
        </v-col>
      </v-row>
      <AuditResults
        v-if="result"
        :verdicts="verdicts"
        :notes="notes"
        :entries="result.image_audit?.audit_results || []"
      />
    </v-card-text>
  </v-card>
</template>

<script setup>
import { computed, inject, ref } from "vue";
import { apiPost, uploadFile } from "../api";
import { t } from "../i18n";
import { formatBytes } from "../utils";
import AuditResults from "./AuditResults.vue";

const notify = inject("notify", () => {});
const prompt = ref("");
const pendingFiles = ref([]);
const uploads = ref([]);
const uploading = ref(false);
const testing = ref(false);
const result = ref(null);

const verdicts = computed(() => {
  const items = [];
  for (const [key, label] of [
    ["prompt_audit", t("audit.promptResult", "提示词审核")],
    ["image_audit", t("audit.imageResult", "图片审核")],
  ]) {
    const verdict = result.value?.[key];
    if (verdict) {
      items.push({ label, allowed: verdict.allowed !== false, reason: verdict.reason || "" });
    }
  }
  return items;
});

const notes = computed(() => {
  const enabled = result.value?.enabled || {};
  const lines = [];
  if (result.value?.prompt_audit && !enabled.prompt_blocked_words && !enabled.prompt_ai) {
    lines.push(
      t(
        "audit.testDisabledPrompt",
        "提示词审核未启用（无屏蔽词且未开启 AI 提示词审核），任何提示词都会通过。",
      ),
    );
  }
  if (result.value?.image_audit && !enabled.image_moderation && !enabled.image_ai) {
    lines.push(
      t(
        "audit.testDisabledImage",
        "图片审核未启用（Moderation 接口审核与 AI 图片审核均关闭），任何图片都会通过。",
      ),
    );
  }
  return lines;
});

async function handleFiles(files) {
  const list = Array.isArray(files) ? files : files ? [files] : [];
  if (!list.length) {
    return;
  }
  uploading.value = true;
  try {
    for (const file of list) {
      uploads.value.push(await uploadFile("page/reference/upload", file));
    }
    notify(t("message.uploaded", "参考图已上传"));
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    uploading.value = false;
    pendingFiles.value = [];
  }
}

async function runTest() {
  const text = prompt.value.trim();
  if (!text && !uploads.value.length) {
    notify(t("audit.testEmpty", "请先填写测试提示词或上传测试图片"), "error");
    return;
  }
  testing.value = true;
  try {
    result.value = await apiPost("page/audit/test", {
      prompt: text,
      reference_tokens: uploads.value.map((item) => item.token),
    });
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    testing.value = false;
  }
}
</script>
