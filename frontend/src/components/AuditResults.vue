<template>
  <div v-if="verdicts.length || notes.length || entries.length" class="audit-grid">
    <div
      v-for="verdict in verdicts"
      :key="verdict.label"
      class="audit-card"
      :class="{ 'audit-card--blocked': !verdict.allowed }"
    >
      <div class="audit-card__head">
        <strong>{{ verdict.label }}</strong>
        <v-chip :color="verdict.allowed ? 'success' : 'error'" size="x-small" variant="tonal">
          {{ verdict.allowed ? t("audit.passed", "通过") : t("audit.blocked", "已拦截") }}
        </v-chip>
      </div>
      <div v-if="verdict.reason" class="audit-note">{{ verdict.reason }}</div>
    </div>

    <div v-for="note in notes" :key="note" class="audit-note audit-grid__wide">
      {{ note }}
    </div>

    <div
      v-for="(entry, index) in entries"
      :key="index"
      class="audit-card"
      :class="{ 'audit-card--blocked': entry.blocked }"
    >
      <div class="audit-card__head">
        <strong>{{ entryTitle(entry) }}</strong>
        <v-chip :color="entry.blocked ? 'error' : 'success'" size="x-small" variant="tonal">
          {{ entry.blocked ? t("audit.blocked", "已拦截") : t("audit.passed", "通过") }}
        </v-chip>
      </div>
      <template v-if="entry.stage === 'ai'">
        <div v-if="entry.reason" class="audit-note">{{ entry.reason }}</div>
      </template>
      <template v-else>
        <div v-for="[name, score] in sortedScores(entry)" :key="name" class="audit-score">
          <span>{{ CATEGORY_LABELS[name] ? `${CATEGORY_LABELS[name]} ${name}` : name }}</span>
          <v-progress-linear
            :model-value="score * 100"
            :color="score >= 0.5 ? 'error' : 'primary'"
            height="6"
            rounded
          />
          <span class="mono">{{ score.toFixed(3) }}</span>
        </div>
        <div v-if="entryNotes(entry)" class="audit-note">{{ entryNotes(entry) }}</div>
        <div v-if="entry.error" class="audit-note text-error">{{ entry.error }}</div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { t } from "../i18n";

defineProps({
  // Per-image audit records from `audit_results`.
  entries: { type: Array, default: () => [] },
  // Overall verdicts: `{label, allowed, reason}`.
  verdicts: { type: Array, default: () => [] },
  // Plain informational lines shown between verdicts and per-image records.
  notes: { type: Array, default: () => [] },
});

const CATEGORY_LABELS = {
  porn: "真人色情",
  hentai: "二次元色情",
  sexy: "性感擦边",
  drawings: "普通绘画",
  neutral: "正常内容",
};

function entryTitle(entry) {
  if (entry.stage === "ai") {
    return t("audit.stageAi", "AI 模型审核");
  }
  return entry.image_name
    ? `#${entry.image_index || "?"} ${entry.image_name}`
    : t("audit.stageModeration", "Moderation 接口审核");
}

function sortedScores(entry) {
  const scores = entry.scores && typeof entry.scores === "object" ? entry.scores : {};
  return Object.entries(scores)
    .map(([name, value]) => [name, Math.max(0, Math.min(1, Number(value) || 0))])
    .sort((a, b) => b[1] - a[1]);
}

function entryNotes(entry) {
  const notes = [];
  if (entry.flagged) {
    notes.push(t("audit.flagged", "服务端标记违规 (flagged)"));
  }
  const hits = Array.isArray(entry.hits) ? entry.hits.filter(Boolean) : [];
  if (hits.length) {
    notes.push(`${t("audit.hits", "命中规则")}: ${hits.join(", ")}`);
  }
  return notes.join(" · ");
}
</script>

<style scoped>
.audit-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}

.audit-grid__wide {
  grid-column: 1 / -1;
}

.audit-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
  padding: 12px 14px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  border-radius: 8px;
}

.audit-card--blocked {
  border-color: rgba(var(--v-theme-error), 0.5);
  background: rgba(var(--v-theme-error), 0.04);
}

.audit-card__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 0.85rem;
}

/* Generated file names have no spaces; let them break so the status chip
   stays inside the card. */
.audit-card__head strong {
  min-width: 0;
  overflow-wrap: anywhere;
}

.audit-card__head .v-chip {
  flex-shrink: 0;
}

.audit-score {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  font-size: 0.78rem;
}

.audit-note {
  font-size: 0.78rem;
  line-height: 1.5;
  color: rgba(var(--v-theme-on-surface), 0.62);
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
