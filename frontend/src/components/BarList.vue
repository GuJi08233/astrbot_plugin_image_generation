<template>
  <div>
    <div v-if="!items.length" class="empty-line">
      {{ emptyText }}
    </div>
    <div v-for="item in items" :key="item.name" class="bar-row">
      <div class="bar-head">
        <span class="bar-name">{{ item.name }}</span>
        <span class="bar-value">
          {{ item.count }}
          <template v-if="item.meta"> · {{ item.meta }}</template>
        </span>
      </div>
      <div class="bar-track">
        <span class="bar-fill" :style="{ width: percent(item.count) }" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  items: { type: Array, default: () => [] },
  emptyText: { type: String, default: "--" },
});

const max = computed(() =>
  Math.max(1, ...props.items.map((item) => Number(item.count) || 0)),
);

const percent = (count) =>
  `${Math.max(3, ((Number(count) || 0) / max.value) * 100)}%`;
</script>

<style scoped>
.bar-row + .bar-row {
  margin-top: 12px;
}

.bar-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 5px;
}

.bar-name {
  font-size: 0.82rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bar-value {
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.55);
  white-space: nowrap;
}

.bar-track {
  height: 6px;
  border-radius: 999px;
  background: rgba(var(--v-theme-on-surface), 0.08);
  overflow: hidden;
}

.bar-fill {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: rgb(var(--v-theme-primary));
}

.empty-line {
  padding: 32px 0;
  text-align: center;
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.45);
}
</style>
