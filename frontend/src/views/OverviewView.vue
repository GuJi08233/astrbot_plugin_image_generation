<template>
  <div>
    <div class="section-head">
      <div>
        <h2 class="section-title">{{ t("overview.statsTitle", "使用统计") }}</h2>
        <div class="section-caption mt-1">{{ rangeCaption }}</div>
      </div>
      <v-select
        v-model="days"
        :items="dayOptions"
        density="compact"
        variant="outlined"
        hide-details
        style="max-width: 168px"
        :loading="loading"
        @update:model-value="load"
      />
    </div>

    <v-row dense>
      <v-col cols="12" sm="6" md="3">
        <StatCard
          :label="t('overview.totalTasks', '总任务')"
          :value="summary.total_tasks ?? '--'"
          :meta="avgDurationText"
          icon="$images"
        />
      </v-col>
      <v-col cols="12" sm="6" md="3">
        <StatCard
          :label="t('overview.successRate', '成功率')"
          :value="`${summary.success_rate ?? 0}%`"
          :meta="successMeta"
          icon="$success"
        />
      </v-col>
      <v-col cols="12" sm="6" md="3">
        <StatCard
          :label="t('overview.totalImages', '生成图片')"
          :value="summary.total_images ?? '--'"
          :meta="imageMeta"
          icon="$image"
        />
      </v-col>
      <v-col cols="12" sm="6" md="3">
        <StatCard
          :label="t('overview.uniqueUsers', '用户数')"
          :value="summary.unique_users ?? '--'"
          :meta="modelMeta"
          icon="$dashboard"
        />
      </v-col>
    </v-row>

    <v-card variant="outlined" class="rounded-lg mt-4">
      <div class="card-title-row">
        <span class="card-title-text">{{ t("overview.dailyTrend", "每日任务量") }}</span>
        <span class="section-caption">{{ trendCaption }}</span>
      </div>
      <v-divider />
      <v-card-text class="pa-4">
        <TrendLineChart
          :points="stats?.trend || []"
          :empty-text="t('overview.noData', '暂无统计数据')"
        />
      </v-card-text>
    </v-card>

    <v-row dense class="mt-1">
      <v-col cols="12" md="6">
        <v-card variant="outlined" class="rounded-lg h-100">
          <div class="card-title-row">
            <span class="card-title-text">{{ t("overview.statusChart", "任务状态") }}</span>
          </div>
          <v-divider />
          <v-card-text class="pa-4">
            <DonutChart
              :items="statusItems"
              :total-label="t('overview.totalTasks', '总任务')"
              :empty-text="t('overview.noData', '暂无统计数据')"
            />
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="6">
        <v-card variant="outlined" class="rounded-lg h-100">
          <div class="card-title-row">
            <span class="card-title-text">{{ t("overview.sourceChart", "任务来源") }}</span>
          </div>
          <v-divider />
          <v-card-text class="pa-4">
            <DonutChart
              :items="stats?.source_distribution || []"
              :total-label="t('overview.totalTasks', '总任务')"
              :empty-text="t('overview.noData', '暂无统计数据')"
            />
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="6">
        <v-card variant="outlined" class="rounded-lg h-100">
          <div class="card-title-row">
            <span class="card-title-text">{{ t("overview.modelChart", "模型使用") }}</span>
          </div>
          <v-divider />
          <v-card-text class="pa-4">
            <BarList :items="modelItems" :empty-text="t('overview.noData', '暂无统计数据')" />
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="6">
        <v-card variant="outlined" class="rounded-lg h-100">
          <div class="card-title-row">
            <span class="card-title-text">{{ t("overview.userChart", "活跃用户") }}</span>
          </div>
          <v-divider />
          <v-card-text class="pa-4">
            <BarList
              :items="stats?.user_distribution || []"
              :empty-text="t('overview.noData', '暂无统计数据')"
            />
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { apiGet } from "../api";
import { t } from "../i18n";
import { formatDuration, statusCssColor, statusLabel } from "../utils";
import StatCard from "../components/StatCard.vue";
import BarList from "../components/BarList.vue";
import TrendLineChart from "../components/TrendLineChart.vue";
import DonutChart from "../components/DonutChart.vue";

const props = defineProps({
  refreshKey: { type: Number, default: 0 },
});

const days = ref(7);
const stats = ref(null);
const loading = ref(false);

const dayOptions = computed(() => [
  { title: t("gallery.days1", "近 1 天"), value: 1 },
  { title: t("gallery.days7", "近 7 天"), value: 7 },
  { title: t("gallery.days30", "近 30 天"), value: 30 },
  { title: t("gallery.daysAll", "全部时间"), value: 0 },
]);

const summary = computed(() => stats.value?.summary || {});
const trendTotal = computed(() =>
  (stats.value?.trend || []).reduce((sum, point) => sum + (point.count || 0), 0),
);

const rangeCaption = computed(() => {
  const current = dayOptions.value.find((item) => item.value === days.value);
  return t("overview.rangeCaption", "{range} · 共 {total} 个任务", {
    range: current?.title || "",
    total: summary.value.total_tasks ?? 0,
  });
});

const trendCaption = computed(() =>
  t("overview.trendCaption", "{granularity} · 区间内 {total} 个任务", {
    granularity:
      stats.value?.trend_granularity === "hour"
        ? t("overview.granularityHour", "按小时")
        : t("overview.granularityDay", "按天"),
    total: trendTotal.value,
  }),
);

const avgDurationText = computed(() =>
  summary.value.avg_duration_seconds
    ? t("overview.avgDuration", "平均耗时 {duration}", {
        duration: formatDuration(summary.value.avg_duration_seconds),
      })
    : "",
);

const successMeta = computed(() =>
  t("overview.successMeta", "成功 {success} / 终态 {terminal}", {
    success: summary.value.success_tasks ?? 0,
    terminal:
      (summary.value.success_tasks ?? 0) +
      (summary.value.failed_tasks ?? 0) +
      (summary.value.cancelled_tasks ?? 0),
  }),
);

const imageMeta = computed(() =>
  t("overview.imageMeta", "可用 {available}", {
    available: summary.value.available_images ?? 0,
  }),
);

const modelMeta = computed(() =>
  t("overview.modelMeta", "模型 {models}", {
    models: summary.value.unique_models ?? 0,
  }),
);

const modelItems = computed(() =>
  (stats.value?.model_distribution || []).map((item) => ({
    name: item.name,
    count: item.count,
    meta: t("overview.modelSuccessRate", "成功率 {rate}%", {
      rate: item.success_rate ?? 0,
    }),
  })),
);

const statusItems = computed(() =>
  (stats.value?.status_distribution || []).map((item) => ({
    name: statusLabel(item.name),
    count: item.count,
    color: statusCssColor(item.name),
  })),
);

async function load() {
  loading.value = true;
  try {
    stats.value = await apiGet("page/stats", { days: days.value });
  } catch {
    stats.value = null;
  } finally {
    loading.value = false;
  }
}

watch(
  () => props.refreshKey,
  () => load(),
);

onMounted(load);
</script>
