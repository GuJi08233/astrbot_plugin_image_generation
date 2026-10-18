<template>
  <div class="donut-wrap">
    <div v-if="!total" class="empty-line">{{ emptyText }}</div>
    <template v-else>
      <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`">
        <circle
          :cx="center"
          :cy="center"
          :r="radius"
          fill="none"
          :stroke-width="thickness"
          class="donut-track"
        />
        <circle
          v-for="segment in segments"
          :key="segment.name"
          :cx="center"
          :cy="center"
          :r="radius"
          fill="none"
          :stroke="segment.color"
          :stroke-width="thickness"
          :stroke-dasharray="`${segment.length} ${circumference - segment.length}`"
          :stroke-dashoffset="segment.offset"
          :transform="`rotate(-90 ${center} ${center})`"
          class="donut-segment"
        >
          <title>{{ segment.name }}: {{ segment.count }}</title>
        </circle>
        <text :x="center" :y="center - 2" text-anchor="middle" class="donut-total">
          {{ total }}
        </text>
        <text :x="center" :y="center + 16" text-anchor="middle" class="donut-caption">
          {{ totalLabel }}
        </text>
      </svg>

      <ul class="donut-legend">
        <li v-for="segment in segments" :key="segment.name">
          <span class="legend-dot" :style="{ background: segment.color }" />
          <span class="legend-name">{{ segment.name }}</span>
          <span class="legend-value">
            {{ segment.count }}
            <span class="legend-percent">{{ segment.percent }}%</span>
          </span>
        </li>
      </ul>
    </template>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  items: { type: Array, default: () => [] },
  size: { type: Number, default: 148 },
  thickness: { type: Number, default: 18 },
  totalLabel: { type: String, default: "总计" },
  emptyText: { type: String, default: "--" },
});

const PALETTE = [
  "rgb(var(--v-theme-primary))",
  "rgb(var(--v-theme-success))",
  "rgb(var(--v-theme-warning))",
  "rgb(var(--v-theme-error))",
  "rgb(var(--v-theme-info))",
  "rgb(var(--v-theme-secondary))",
  "rgba(var(--v-theme-on-surface), 0.45)",
];

const center = computed(() => props.size / 2);
const radius = computed(() => props.size / 2 - props.thickness / 2 - 2);
const circumference = computed(() => 2 * Math.PI * radius.value);

const total = computed(() =>
  props.items.reduce((sum, item) => sum + (Number(item.count) || 0), 0),
);

const segments = computed(() => {
  let accumulated = 0;
  return props.items
    .filter((item) => Number(item.count) > 0)
    .map((item, index) => {
      const count = Number(item.count) || 0;
      const length = (count / total.value) * circumference.value;
      const segment = {
        name: item.name,
        count,
        color: item.color || PALETTE[index % PALETTE.length],
        length,
        offset: -accumulated,
        percent: Math.round((count / total.value) * 1000) / 10,
      };
      accumulated += length;
      return segment;
    });
});
</script>

<style scoped>
.donut-wrap {
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}

.donut-track {
  stroke: rgba(var(--v-theme-on-surface), 0.08);
}

.donut-segment {
  transition: stroke-dasharray 0.3s ease;
}

.donut-total {
  fill: rgb(var(--v-theme-on-surface));
  font-size: 20px;
  font-weight: 700;
}

.donut-caption {
  fill: rgba(var(--v-theme-on-surface), 0.5);
  font-size: 10px;
}

.donut-legend {
  list-style: none;
  margin: 0;
  padding: 0;
  flex: 1 1 140px;
  min-width: 140px;
}

.donut-legend li {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  font-size: 0.8rem;
}

.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex: none;
}

.legend-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.legend-value {
  color: rgba(var(--v-theme-on-surface), 0.7);
  white-space: nowrap;
}

.legend-percent {
  color: rgba(var(--v-theme-on-surface), 0.45);
  margin-left: 4px;
}

.empty-line {
  padding: 48px 0;
  width: 100%;
  text-align: center;
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.45);
}
</style>
