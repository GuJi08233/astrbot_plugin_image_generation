<template>
  <v-dialog
    :model-value="modelValue"
    max-width="1000"
    @update:model-value="emitClose"
  >
    <v-card class="rounded-lg">
      <v-card-title class="text-h3 pa-4 pb-0 pl-6">
        {{ title }}
      </v-card-title>
      <v-card-text
        class="d-flex justify-center align-center"
        style="min-height: 240px"
      >
        <img v-if="src" :src="src" alt="" class="lightbox-image" />
        <span v-else class="section-caption">{{ emptyText }}</span>
      </v-card-text>
      <v-card-actions class="pa-4 pt-0">
        <v-spacer />
        <v-btn
          v-if="src && downloadEndpoint"
          variant="tonal"
          prepend-icon="$download"
          @click="emit('download')"
        >
          {{ downloadText }}
        </v-btn>
        <v-btn variant="text" @click="emitClose(false)">{{ closeText }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
defineProps({
  modelValue: { type: Boolean, default: false },
  src: { type: String, default: "" },
  title: { type: String, default: "" },
  emptyText: { type: String, default: "--" },
  downloadEndpoint: { type: String, default: "" },
  downloadText: { type: String, default: "下载" },
  closeText: { type: String, default: "关闭" },
});

const emit = defineEmits(["update:modelValue", "download"]);

const emitClose = (value) => emit("update:modelValue", value);
</script>
