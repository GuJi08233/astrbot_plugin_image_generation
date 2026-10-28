<template>
  <v-expansion-panels variant="accordion">
    <v-expansion-panel>
      <v-expansion-panel-title>{{ t("novelai.title", "NovelAI 高级设置") }}</v-expansion-panel-title>
      <v-expansion-panel-text>
        <div class="d-flex flex-column ga-4 pt-2">
          <v-select
            v-model="state.mode"
            :items="modeOptions"
            :label="t('novelai.mode', '生成模式')"
            variant="outlined"
            density="comfortable"
            hide-details
          />

          <v-textarea
            v-model="state.parametersText"
            :label="t('novelai.parameters', '本次采样与多角色参数（JSON）')"
            :hint="t('novelai.parametersHint', '留空继承供应商默认值。支持 steps、sampler、scale、seed、noise_schedule、cfg_rescale、sm、sm_dyn、dynamic_thresholding、variety_boost 和 characters。角色位置为 A1–E5 或 [x,y]，最多 6 人。宽高使用上方输入框。')"
            placeholder='{"steps":28,"scale":5,"sampler":"k_euler_ancestral","characters":[{"prompt":"1girl, red hair","negative_prompt":"hat","position":"B3"},{"prompt":"1boy, blue hair","position":"D3"}]}'
            rows="5"
            auto-grow
            spellcheck="false"
            variant="outlined"
            persistent-hint
            class="mono-input"
          />

          <div v-if="['auto', 'infill'].includes(state.mode)">
            <v-file-input
              v-model="pendingMask"
              :label="t('novelai.mask', '局部重绘蒙版')"
              :hint="t('novelai.maskHint', '白色区域重绘，黑色区域保留。上传与源图尺寸一致的蒙版；局部重绘需要一张源图。')"
              :loading="uploading === 'mask'"
              :disabled="Boolean(uploading)"
              accept="image/*"
              variant="outlined"
              density="comfortable"
              persistent-hint
              prepend-icon=""
              prepend-inner-icon="$image"
              @update:model-value="(files) => upload(files, 'mask')"
            />
            <v-chip
              v-if="state.mask"
              size="small"
              closable
              class="mt-2"
              @click:close="state.mask = null"
            >
              {{ state.mask.filename }}
            </v-chip>
          </div>

          <div>
            <v-file-input
              v-model="pendingReferences"
              :label="t('novelai.characterReferences', '角色精确参考')"
              :hint="t('novelai.characterReferencesHint', '为受支持的 NAI4.5 模型上传参考图，逐张设置类型、强度和保真度。不能与 Vibe 同时使用；这些图片不会作为图生图源图。')"
              :loading="uploading === 'reference'"
              :disabled="Boolean(uploading)"
              accept="image/*"
              multiple
              variant="outlined"
              density="comfortable"
              persistent-hint
              prepend-icon=""
              prepend-inner-icon="$image"
              @update:model-value="(files) => upload(files, 'reference')"
            />
            <div
              v-for="(reference, index) in state.references"
              :key="`${reference.token}-${index}`"
              class="reference-row mt-3"
            >
              <v-chip size="small" closable @click:close="state.references.splice(index, 1)">
                {{ reference.filename }}
              </v-chip>
              <v-select
                v-model="reference.type"
                :items="referenceTypes"
                :label="t('novelai.referenceType', '参考类型')"
                variant="outlined"
                density="compact"
                hide-details
              />
              <v-text-field
                v-for="key in ['strength', 'fidelity']"
                :key="key"
                v-model.number="reference[key]"
                type="number"
                min="0"
                max="1"
                step="0.05"
                :label="key === 'strength' ? t('novelai.strength', '强度') : t('novelai.fidelity', '保真度')"
                variant="outlined"
                density="compact"
                hide-details
              />
            </div>
          </div>

          <v-textarea
            v-model="state.vibeText"
            :label="t('novelai.vibe', '已编码 Vibe')"
            :hint="t('novelai.vibeHint', 'V4 系列必须提供已编码数据，最多 16 项。此处不会调用付费编码接口。V3 图片参考可在高级 JSON 的 vibe_references 中填写 image、information_extracted 和 strength。')"
            :placeholder="t('novelai.vibePlaceholder', '粘贴已编码 Base64，或 [{&quot;encoding&quot;:&quot;...&quot;,&quot;information_extracted&quot;:1,&quot;strength&quot;:0.6}]')"
            rows="3"
            spellcheck="false"
            variant="outlined"
            persistent-hint
            class="mono-input"
            @update:model-value="clearVibeImport"
          />

          <v-file-input
            v-model="pendingVibe"
            :label="t('novelai.vibeFile', '导入 Vibe（.naiv4vibe / JSON / Base64 文本）')"
            accept="text/plain,application/json,.txt,.json,.naiv4vibe"
            variant="outlined"
            density="comfortable"
            hide-details
            prepend-icon=""
            prepend-inner-icon="$upload"
            @update:model-value="importVibe"
          />

          <v-select
            v-if="state.vibeImport && state.vibeImport.variants.length > 1"
            v-model="state.vibeVariant"
            :items="variantOptions"
            :label="t('novelai.vibeVariant', '选择导入文件中的编码变体')"
            :placeholder="t('novelai.vibeVariantRequired', '请选择一个编码变体后再生成')"
            variant="outlined"
            density="comfortable"
            hide-details
            @update:model-value="selectVariant"
          />
        </div>
      </v-expansion-panel-text>
    </v-expansion-panel>
  </v-expansion-panels>
</template>

<script setup>
import { computed, inject, ref } from "vue";
import { uploadFile } from "../api";
import { t } from "../i18n";
import { formatVibeVariant, parseVibeFile } from "../novelai";

const props = defineProps({
  // Bare model id of the selected NovelAI model, used to match Vibe imports.
  model: { type: String, default: "" },
});

// Owned by GenerateView so the submit handler can build the request from it.
const state = defineModel({ type: Object, required: true });

const notify = inject("notify", () => {});
const pendingMask = ref([]);
const pendingReferences = ref([]);
const pendingVibe = ref([]);
const uploading = ref("");

const modeOptions = computed(() => [
  { title: t("novelai.modeAuto", "自动（根据源图和蒙版）"), value: "auto" },
  { title: t("novelai.modeText", "文生图"), value: "generate" },
  { title: t("novelai.modeImage", "图生图"), value: "img2img" },
  { title: t("novelai.modeInpaint", "局部重绘"), value: "infill" },
]);

const referenceTypes = computed(() =>
  ["character", "style", "character&style"].map((value) => ({
    title: t(`novelai.type.${value}`, value),
    value,
  })),
);

const variantOptions = computed(() =>
  (state.value.vibeImport?.variants || []).map((entry, index) => ({
    title: `${t("novelai.vibeInformation", "信息提取量")} ${entry.information_extracted} · ${entry.key.slice(0, 12)}`,
    value: String(index),
  })),
);

async function upload(files, role) {
  const list = Array.isArray(files) ? files : files ? [files] : [];
  if (!list.length) {
    return;
  }
  if (role === "reference" && list.length + state.value.references.length > 16) {
    notify(t("novelai.referencesTooMany", "角色精确参考最多允许 16 项（包括高级 JSON 中的参考）。"), "error");
    pendingReferences.value = [];
    return;
  }
  uploading.value = role;
  try {
    for (const file of list) {
      const result = await uploadFile("page/reference/upload", file);
      if (role === "mask") {
        state.value.mask = result;
      } else {
        state.value.references.push({ ...result, type: "character", strength: 1, fidelity: 1 });
      }
    }
    notify(t("message.uploaded", "参考图已上传"));
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    uploading.value = "";
    pendingMask.value = [];
    pendingReferences.value = [];
  }
}

function clearVibeImport() {
  // Hand-edited text no longer matches the imported variants.
  state.value.vibeImport = null;
  state.value.vibeVariant = "";
}

async function importVibe(files) {
  const file = Array.isArray(files) ? files[0] : files;
  if (!file) {
    return;
  }
  try {
    if (file.size > 16 * 1024 * 1024) {
      throw new Error(t("novelai.vibeTooLarge", "已编码 Vibe 文本不能超过 16 MiB。"));
    }
    const { text, vibeImport } = parseVibeFile(await file.text(), file.name, props.model, t);
    state.value.vibeText = text;
    state.value.vibeImport = vibeImport;
    state.value.vibeVariant = "";
  } catch (error) {
    notify(error?.message || t("error.generic", "操作失败"), "error");
  } finally {
    pendingVibe.value = [];
  }
}

function selectVariant(value) {
  const entry = state.value.vibeImport?.variants[Number(value)];
  state.value.vibeText = value === "" || !entry ? "" : formatVibeVariant(entry);
}
</script>

<style scoped>
.reference-row {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(0, 1.2fr) minmax(0, 0.8fr) minmax(0, 0.8fr);
  align-items: center;
  gap: 8px;
}

.mono-input :deep(textarea) {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 12px;
}

@media (max-width: 600px) {
  .reference-row {
    grid-template-columns: 1fr 1fr;
  }
}
</style>
