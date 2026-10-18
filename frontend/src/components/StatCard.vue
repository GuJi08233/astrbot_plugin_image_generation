<template>
  <v-card variant="outlined" class="stat-card rounded-lg">
    <v-card-text class="d-flex flex-column ga-2">
      <div class="d-flex align-center justify-space-between">
        <span class="stat-label">{{ label }}</span>
        <span v-if="icon" class="stat-icon">
          <v-icon :icon="icon" size="16" />
        </span>
      </div>
      <div class="stat-value">{{ value }}</div>
      <div class="stat-meta">{{ meta || "&nbsp;" }}</div>
    </v-card-text>
  </v-card>
</template>

<script setup>
defineProps({
  label: { type: String, default: "" },
  value: { type: [String, Number], default: "--" },
  meta: { type: String, default: "" },
  icon: { type: String, default: "" },
});
</script>

<style scoped>
.stat-card {
  height: 100%;
}

.stat-label {
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.62);
}

.stat-value {
  font-size: 1.7rem;
  font-weight: 700;
  line-height: 1.15;
}

.stat-meta {
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.5);
}

.stat-icon {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: rgba(var(--v-theme-primary), 0.12);
  color: rgb(var(--v-theme-primary));
}
</style>
