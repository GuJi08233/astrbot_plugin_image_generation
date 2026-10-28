<template>
  <v-dialog v-model="show" max-width="760">
    <v-card class="rounded-lg">
      <v-card-title class="text-h3 pa-4 pb-0 pl-6">
        {{ t("audit.reauditTitle", "复审结果") }}
      </v-card-title>
      <v-card-text class="pa-6">
        <div v-if="loading" class="d-flex justify-center pa-6">
          <v-progress-circular indeterminate />
        </div>
        <AuditResults
          v-else-if="result"
          :verdicts="verdicts"
          :notes="notes"
          :entries="result.image_audit?.audit_results || []"
        />
      </v-card-text>
      <v-card-actions class="pa-4 pt-0">
        <v-spacer />
        <v-btn variant="text" @click="show = false">{{ t("lightbox.close", "关闭") }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { computed, inject, ref } from "vue";
import { apiPost } from "../api";
import { t } from "../i18n";
import AuditResults from "./AuditResults.vue";

const notify = inject("notify", () => {});
const show = ref(false);
const loading = ref(false);
const result = ref(null);

const verdicts = computed(() => {
  const verdict = result.value?.image_audit || {};
  return [
    {
      label: t("audit.imageResult", "图片审核"),
      allowed: verdict.allowed !== false,
      reason: verdict.reason || "",
    },
  ];
});

const notes = computed(() => {
  const lines = [
    t(
      "audit.reauditCount",
      "本次复审 {count} 张图片，按当前审核配置执行，不影响任务原有审核记录。",
      { count: result.value?.audited_count || 0 },
    ),
  ];
  const enabled = result.value?.enabled || {};
  if (!enabled.image_moderation && !enabled.image_ai) {
    lines.push(
      t(
        "audit.testDisabledImage",
        "图片审核未启用（Moderation 接口审核与 AI 图片审核均关闭），任何图片都会通过。",
      ),
    );
  }
  return lines;
});

/**
 * Re-run the image audit for a task with the current audit configuration.
 *
 * @param {string} taskId Generation task id.
 * @param {number} [imageIndex] 1-based image index; omit to audit every image.
 */
async function open(taskId, imageIndex) {
  if (!taskId) {
    return;
  }
  result.value = null;
  loading.value = true;
  show.value = true;
  try {
    result.value = await apiPost(
      `page/tasks/${encodeURIComponent(taskId)}/reaudit`,
      imageIndex ? { image_index: Number(imageIndex) } : {},
    );
  } catch (error) {
    show.value = false;
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    loading.value = false;
  }
}

defineExpose({ open });
</script>
