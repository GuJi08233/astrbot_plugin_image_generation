import { createApp } from "vue";
import "vuetify/styles";
import App from "./App.vue";
import { vuetify } from "./theme";

createApp(App).use(vuetify).mount("#app");
