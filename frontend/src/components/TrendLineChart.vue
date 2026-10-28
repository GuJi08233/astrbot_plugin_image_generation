<template>
  <div ref="container" class="line-chart">
    <div v-if="!points.length" class="empty-line">{{ emptyText }}</div>
    <svg
      v-else-if="width > 0"
      :width="width"
      :height="height"
      class="line-chart__svg"
      role="img"
    >
      <g class="grid">
        <template v-for="line in gridLines" :key="line.value">
          <line
            :x1="pad.left"
            :x2="width - pad.right"
            :y1="line.y"
            :y2="line.y"
            class="grid-line"
          />
          <text :x="pad.left - 8" :y="line.y + 4" text-anchor="end" class="axis-text">
            {{ line.value }}
          </text>
        </template>
      </g>

      <path :d="areaPath" class="area" />
      <path :d="linePath" class="line" />

      <template v-for="(point, index) in plotted" :key="point.key">
        <circle :cx="point.x" :cy="point.y" r="3" class="dot" />
        <text
          v-if="showLabel(index)"
          :x="point.x"
          :y="height - 8"
          text-anchor="middle"
          class="axis-text"
        >
          {{ point.label }}
        </text>
      </template>
    </svg>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";

const props = defineProps({
  points: { type: Array, default: () => [] },
  height: { type: Number, default: 208 },
  emptyText: { type: String, default: "--" },
});

const container = ref(null);
const width = ref(0);

const pad = { top: 16, right: 14, bottom: 26, left: 38 };

let observer = null;

// ResizeObserver notifications are throttled while the page/tab is hidden, so
// the width is also measured synchronously on mount and on window resize.
function measure() {
  const element = container.value;
  if (!element) {
    return;
  }
  const next = element.clientWidth || element.getBoundingClientRect().width;
  if (next > 0) {
    width.value = Math.floor(next);
  }
}

onMounted(() => {
  measure();
  nextTick(measure);
  window.requestAnimationFrame?.(measure);
  window.addEventListener("resize", measure);
  if (container.value && typeof ResizeObserver !== "undefined") {
    observer = new ResizeObserver(measure);
    observer.observe(container.value);
  }
});

onBeforeUnmount(() => {
  observer?.disconnect();
  observer = null;
  window.removeEventListener("resize", measure);
});

const niceMax = computed(() => {
  const max = Math.max(1, ...props.points.map((p) => Number(p.count) || 0));
  const magnitude = 10 ** Math.floor(Math.log10(max));
  const step = [1, 2, 5, 10].find((m) => max / (magnitude * m) <= 1) || 10;
  return magnitude * step;
});

const innerWidth = computed(() => Math.max(1, width.value - pad.left - pad.right));
const innerHeight = computed(() => Math.max(1, props.height - pad.top - pad.bottom));

const xAt = (index) => {
  if (props.points.length <= 1) {
    return pad.left + innerWidth.value / 2;
  }
  return pad.left + (innerWidth.value * index) / (props.points.length - 1);
};

const yAt = (count) =>
  pad.top + innerHeight.value * (1 - (Number(count) || 0) / niceMax.value);

const plotted = computed(() =>
  props.points.map((point, index) => ({
    ...point,
    x: xAt(index),
    y: yAt(point.count),
  })),
);

const linePath = computed(() => {
  const points = plotted.value;
  if (!points.length) {
    return "";
  }
  if (points.length === 1) {
    return `M${points[0].x},${points[0].y}`;
  }
  // Catmull-Rom → cubic Bezier keeps the curve smooth without overshooting
  // below the chart baseline.
  const tension = 0.18;
  const baseline = pad.top + innerHeight.value;
  const clampY = (value) => Math.min(baseline, Math.max(pad.top, value));
  let path = `M${points[0].x},${points[0].y}`;
  for (let index = 0; index < points.length - 1; index += 1) {
    const p0 = points[index - 1] || points[index];
    const p1 = points[index];
    const p2 = points[index + 1];
    const p3 = points[index + 2] || p2;
    const cp1x = p1.x + (p2.x - p0.x) * tension;
    const cp1y = clampY(p1.y + (p2.y - p0.y) * tension);
    const cp2x = p2.x - (p3.x - p1.x) * tension;
    const cp2y = clampY(p2.y - (p3.y - p1.y) * tension);
    path += ` C${cp1x},${cp1y} ${cp2x},${cp2y} ${p2.x},${p2.y}`;
  }
  return path;
});

const areaPath = computed(() => {
  if (!plotted.value.length) {
    return "";
  }
  const first = plotted.value[0];
  const last = plotted.value[plotted.value.length - 1];
  const baseline = pad.top + innerHeight.value;
  return `${linePath.value} L${last.x},${baseline} L${first.x},${baseline} Z`;
});

const gridLines = computed(() => {
  const segments = 4;
  return Array.from({ length: segments + 1 }, (_, index) => {
    const value = (niceMax.value / segments) * (segments - index);
    return {
      value: Math.round(value * 10) / 10,
      y: pad.top + (innerHeight.value / segments) * index,
    };
  });
});

const labelStep = computed(() =>
  Math.max(1, Math.ceil(props.points.length / Math.max(2, Math.floor(innerWidth.value / 64)))),
);

const showLabel = (index) =>
  index % labelStep.value === 0 || index === props.points.length - 1;
</script>

<style scoped>
.line-chart {
  width: 100%;
  min-height: 120px;
}

.line-chart__svg {
  display: block;
  overflow: visible;
}

.grid-line {
  stroke: rgba(var(--v-theme-on-surface), 0.1);
  stroke-width: 1;
}

.axis-text {
  fill: rgba(var(--v-theme-on-surface), 0.5);
  font-size: 10px;
}

.area {
  fill: rgba(var(--v-theme-primary), 0.14);
}

.line {
  fill: none;
  stroke: rgb(var(--v-theme-primary));
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.dot {
  fill: rgb(var(--v-theme-primary));
  stroke: rgb(var(--v-theme-surface));
  stroke-width: 1.5;
}

.empty-line {
  padding: 56px 0;
  text-align: center;
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.45);
}
</style>
