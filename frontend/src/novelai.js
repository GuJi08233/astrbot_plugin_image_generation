/**
 * NovelAI request helpers for the Generate page.
 *
 * Kept free of Vue and bridge imports so `tests/test_page_novelai.mjs` can run
 * them under `node:test`. Every function receives the page translator `t` and
 * throws `Error` with a translated message when the input is rejected.
 */

export const NOVELAI_MAX_PIXELS = 3145728;
const MAX_PARAMETERS_CHARS = 65536;
const MAX_VIBE_CHARS = 16 * 1024 * 1024;
const LIST_LIMITS = [
  ["characters", 6],
  ["character_references", 16],
  ["vibe_references", 16],
];
// `.naiv4vibe` files group encodings by model family.
const VIBE_MODEL_KEYS = {
  "nai-diffusion-4-5-full": "v4-5full",
  "nai-diffusion-4-5-curated": "v4-5curated",
  "nai-diffusion-4-full": "v4full",
  "nai-diffusion-4-curated": "v4curated",
  "nai-diffusion-4-curated-preview": "v4curated",
};

const isPlainObject = (value) =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

/**
 * Check one manual NovelAI pixel size.
 *
 * @param {number} width Requested width in pixels.
 * @param {number} height Requested height in pixels.
 * @returns {boolean} Whether both sides are 64-multiples in 64–2048 and the
 *   total area stays within the NovelAI limit.
 */
export function isValidPixelSize(width, height) {
  return (
    [width, height].every(
      (value) =>
        Number.isInteger(value) && value >= 64 && value <= 2048 && value % 64 === 0,
    ) && width * height <= NOVELAI_MAX_PIXELS
  );
}

/**
 * Build the request-scoped `novelai` object and role-specific upload fields.
 *
 * @param {object} input Page state.
 * @param {string} input.mode `auto`, `generate`, `img2img` or `infill`.
 * @param {string} input.parametersText Optional advanced JSON object text.
 * @param {{token: string}|null} input.mask Uploaded inpainting mask.
 * @param {Array<{token: string, type: string, strength: number, fidelity: number}>} input.references
 *   Uploaded precise references.
 * @param {string} input.vibeText Encoded Vibe Base64 or JSON text.
 * @param {{model: string, variants: object[]}|null} input.vibeImport Parsed `.naiv4vibe` import.
 * @param {string} input.vibeVariant Selected import variant index, or "".
 * @param {string} input.model Bare model id of the selected NovelAI model.
 * @param {object} payload Page generate payload; `reference_tokens` may be
 *   cleared and `novelai_mask_token` / `novelai_character_references` added.
 * @param {(key: string, fallback?: string, vars?: object) => string} t Translator.
 * @returns {object|null} NovelAI options, or null when nothing overrides defaults.
 */
export function buildNovelaiOptions(input, payload, t) {
  const fail = (key, fallback, vars) => {
    throw new Error(t(`novelai.${key}`, fallback, vars));
  };
  if (input.vibeImport) {
    if (input.model !== input.vibeImport.model) {
      fail("vibeModelMismatch", "导入文件没有当前模型的编码，或导入后更换了模型。请为当前模型重新导入。");
    }
    if (input.vibeImport.variants.length > 1 && input.vibeVariant === "") {
      fail("vibeVariantRequired", "请选择一个编码变体后再生成");
    }
  }
  const raw = String(input.parametersText || "").trim();
  if (raw.length > MAX_PARAMETERS_CHARS) {
    fail("parametersTooLarge", "高级参数 JSON 不能超过 65,536 字符。");
  }
  let options = {};
  if (raw) {
    try {
      options = JSON.parse(raw);
    } catch {
      fail("invalidJson", "高级参数必须是有效的 JSON 对象，i2i 和 inpaint 也必须是对象。");
    }
    if (!isPlainObject(options)) {
      fail("invalidJson", "高级参数必须是有效的 JSON 对象，i2i 和 inpaint 也必须是对象。");
    }
  }
  const mode = input.mode || "auto";
  if (mode !== "auto") {
    if (options.action && options.action !== mode) {
      fail("modeConflict", "生成模式与图生图或局部重绘参数冲突，请检查 JSON 和蒙版。");
    }
    options.action = mode;
    if (mode === "img2img") options.i2i ??= {};
    if (mode === "infill") options.inpaint ??= {};
  }
  for (const key of ["i2i", "inpaint"]) {
    if (key in options && !isPlainObject(options[key])) {
      fail("invalidJson", "高级参数必须是有效的 JSON 对象，i2i 和 inpaint 也必须是对象。");
    }
  }
  if (input.mask && ["auto", "infill"].includes(mode)) {
    payload.novelai_mask_token = input.mask.token;
    options.inpaint ??= {};
  }
  if (
    ("i2i" in options && "inpaint" in options) ||
    (options.action === "generate" && ("i2i" in options || "inpaint" in options)) ||
    (options.action === "img2img" && "inpaint" in options) ||
    (options.action === "infill" && "i2i" in options)
  ) {
    fail("modeConflict", "生成模式与图生图或局部重绘参数冲突，请检查 JSON 和蒙版。");
  }
  if ("inpaint" in options && !options.inpaint.mask && !payload.novelai_mask_token) {
    fail("maskRequired", "局部重绘需要上传蒙版，或在高级 JSON 的 inpaint.mask 中指定蒙版。");
  }
  // Text-to-image must not turn an uploaded source into an edit.
  if (options.action === "generate") payload.reference_tokens = [];
  if (payload.reference_tokens.length > 1) {
    fail("oneSource", "NovelAI 图生图或局部重绘最多使用一张源图，蒙版也必须单独上传一张。");
  }
  const vibeText = String(input.vibeText || "").trim();
  if (vibeText.length > MAX_VIBE_CHARS) {
    fail("vibeTooLarge", "已编码 Vibe 文本不能超过 16 MiB。");
  }
  if (vibeText) {
    let references;
    try {
      references = ["[", "{"].includes(vibeText[0]) ? JSON.parse(vibeText) : vibeText;
    } catch {
      fail("invalidVibe", "Vibe 必须是已编码 Base64、包含 encoding 的 JSON 对象或对象数组。");
    }
    if (!Array.isArray(references)) references = [references];
    references = references.map((entry) =>
      typeof entry === "string"
        ? { encoding: entry, information_extracted: 1, strength: 0.6 }
        : entry,
    );
    if (!Array.isArray(options.vibe_references ?? [])) {
      fail("invalidVibe", "Vibe 必须是已编码 Base64、包含 encoding 的 JSON 对象或对象数组。");
    }
    options.vibe_references = [...(options.vibe_references || []), ...references];
  }
  for (const [key, maximum] of LIST_LIMITS) {
    if (!(key in options)) continue;
    if (
      !Array.isArray(options[key]) ||
      options[key].length > maximum ||
      options[key].some((entry) => !isPlainObject(entry))
    ) {
      fail("invalidList", "{name} 必须是最多 {maximum} 项的对象数组。", { name: key, maximum });
    }
  }
  const references = input.references || [];
  if ((options.character_references?.length || 0) + references.length > 16) {
    fail("referencesTooMany", "角色精确参考最多允许 16 项（包括高级 JSON 中的参考）。");
  }
  if (
    (options.character_references?.length || references.length) &&
    options.vibe_references?.length
  ) {
    fail("referenceVibeConflict", "角色精确参考和 Vibe 不能同时使用。");
  }
  if (references.length) {
    payload.novelai_character_references = references.map(
      ({ token, type, strength, fidelity }) => ({ token, type, strength, fidelity }),
    );
  }
  return Object.keys(options).length ? options : null;
}

/**
 * Parse an imported Vibe file without ever treating embedded images as features.
 *
 * @param {string} text File contents.
 * @param {string} filename Original file name, used to detect `.naiv4vibe`.
 * @param {string} model Bare model id of the selected NovelAI model.
 * @param {(key: string, fallback?: string, vars?: object) => string} t Translator.
 * @returns {{text: string, vibeImport: {model: string, variants: object[]}|null}}
 *   Text for the Vibe field and the parsed variants when the file is a
 *   `novelai-vibe-transfer` export; a multi-variant import leaves `text` empty
 *   until a variant is chosen.
 */
export function parseVibeFile(text, filename, model, t) {
  const isVibeFile = String(filename || "").toLowerCase().endsWith(".naiv4vibe");
  const invalidFile = () => {
    throw new Error(
      t("novelai.invalidVibeFile", "无法读取该 Vibe 文件：需要受支持的格式和有效的编码变体。"),
    );
  };
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    if (isVibeFile) invalidFile();
  }
  if (data?.identifier !== "novelai-vibe-transfer") {
    if (isVibeFile) invalidFile();
    return { text, vibeImport: null };
  }
  if (data.version !== 1 || data.type !== "image" || !isPlainObject(data.encodings)) {
    invalidFile();
  }
  const bucket = data.encodings[VIBE_MODEL_KEYS[model]];
  if (!VIBE_MODEL_KEYS[model] || !isPlainObject(bucket)) {
    throw new Error(
      t("novelai.vibeModelMismatch", "导入文件没有当前模型的编码，或导入后更换了模型。请为当前模型重新导入。"),
    );
  }
  const strength = data.importInfo?.strength ?? 0.6;
  if (!Number.isFinite(strength) || strength < 0 || strength > 1) invalidFile();
  const variants = Object.entries(bucket).map(([key, entry]) => {
    const information = entry?.params?.information_extracted;
    if (
      typeof entry?.encoding !== "string" ||
      !entry.encoding ||
      !Number.isFinite(information) ||
      information <= 0 ||
      information > 1
    ) {
      invalidFile();
    }
    return { key, encoding: entry.encoding, information_extracted: information, strength };
  });
  if (!variants.length || variants.length > 64) invalidFile();
  const vibeImport = { model, variants };
  return {
    text: variants.length === 1 ? formatVibeVariant(variants[0]) : "",
    vibeImport,
  };
}

/**
 * Format one imported Vibe variant as the JSON text sent with the request.
 *
 * @param {{key: string, encoding: string, information_extracted: number, strength: number}} variant
 * @returns {string} A one-item reference array as pretty JSON.
 */
export function formatVibeVariant(variant) {
  const { key: _key, ...reference } = variant;
  return JSON.stringify([reference], null, 2);
}
