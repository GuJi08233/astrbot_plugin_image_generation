# NovelAI 原生 JSON 接口配置

`novelai` 适配器使用 NovelAI 原生请求结构调用 `POST /v1/nai/generate-image`，解析同步 JSON 响应中的 `images` Base64 图片。默认服务地址为 `https://nai.rinko.ai`，也可接入支持相同请求和响应格式的服务。

这个接口与 `/native/ai/generate-image` 使用相同的原生请求结构，但返回 JSON。图生图由 `action: "img2img"` 和 `parameters.image` 调用，不需要处理 ZIP 或 MessagePack。

当前支持：

- 文生图和单张参考图图生图。
- 自动组装 V4 / V4.5 的 `v4_prompt` 与 `v4_negative_prompt`。
- 宽度、高度、步数、采样器、CFG、种子、负向提示词、噪声调度和 CFG 重缩放。
- 图生图重绘强度和噪声。
- 插件统一的多图任务、队列、参考图校验、审核、额度和结果发送流程。
- Bearer Token、代理、请求超时和幂等键。

当前不支持局部重绘、多角色提示词、角色精确参考、Vibe、ZIP 返回值、MessagePack 或 SSE 流式生成。质量标签和 UC 预设不会自动追加；需要的标签应直接写入提示词或预设。图生图仅支持一张参考图，多张参考图会明确报错。

## 快速配置

1. 在“图像模型供应商”中新增 **NovelAI 原生 JSON 接口**。
2. 供应商名称可保留 `NovelAI`，接口地址保留 `https://nai.rinko.ai`。
3. 在“API 密钥”中填写该站点的令牌，例如 `ynai-…`。
4. 根据账号权限维护可用模型。默认提供 `nai-diffusion-4-5-full` 和 `nai-diffusion-4-5-curated`。
5. 在“生图模型”或 `/生图模型` 中选择 `NovelAI/nai-diffusion-4-5-full`。
6. 初次使用可保留默认宽高 `832×1216`、超时 `330` 秒和最大尝试次数 `1`。

默认启用“文生图”“图生图”和“宽高比”。全局“默认分辨率”及单次请求的 `1K` / `2K` / `4K` 分辨率不参与 NovelAI 尺寸计算。

## 文生图和图生图

没有参考图时，`/生图 1girl, masterpiece, best quality` 使用 `action: "generate"`。V4 系列模型建议使用模型支持的英文标签和提示词。

图生图可以直接发送一张图片并输入 `/生图 <修改要求>`，也可以引用一条包含一张图片的消息。插件会收集参考图并将它编码为不带 `data:` 前缀的 Base64，发送 `action: "img2img"`、`parameters.image`、`parameters.strength` 与 `parameters.noise`。

参考图来源仍包括当前消息、引用消息、@ 用户头像和人设图片。请确保去重后的参考图合计只有一张；例如消息图片与另一张人设参考图同时存在时，会超过此适配器的上限。不会静默丢弃其中一张图。

默认重绘强度为 `0.7`。减小该值通常更接近原图，增大通常使重绘变化更大；噪声默认 `0.0`。参考图通过插件既有的图片安全校验后，会校正 EXIF 方向、转换为 RGB PNG，并按目标宽高等比缩放，以白色填充不足的区域；发送的是处理后的图片，避免不同原图尺寸导致请求不匹配。

单个插件任务仍可以生成多张图片。每个子请求发送 `n_samples: 1`，拥有独立的幂等键；文档中服务端支持的 `n_samples: 1–8` 不作为额外批量配置开放。

## 参数与尺寸

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| 默认图片宽度 | `832` | 不指定宽高比时使用。 |
| 默认图片高度 | `1216` | 不指定宽高比时使用。 |
| 采样步数 | `28` | 范围 `1–50`。 |
| 采样器 | `k_euler_ancestral` | 填写服务与模型支持的原生采样器名称。 |
| 提示词遵循强度（CFG） | `5.0` | 范围 `0–10`。 |
| 随机种子 | `-1` | 每个子请求随机选择种子；非负整数表示固定种子，重试保持不变。 |
| 负向提示词 | 空 | 自动用于普通负向提示词和 V4 负向结构。 |
| 噪声调度 | `karras` | 需与模型、采样器兼容。 |
| CFG 重缩放 | `0.0` | 范围 `0–1`。 |
| 图生图重绘强度 | `0.7` | 范围 `0–1`，仅图生图发送。 |
| 图生图噪声 | `0.0` | 范围 `0–1`，仅图生图发送。 |

宽高必须满足：

- 每边 `64–2048` 像素，且是 `64` 的倍数。
- 总面积不超过 `3,145,728` 像素。

不指定宽高比时使用供应商的默认宽高。指定宽高比时，适配器以配置的 `width×height` 面积重新计算宽高，四舍五入到 `64` 的倍数，因此实际比例可能略有偏差。换算结果超过边长或面积限制会在生成前报错，请调整默认宽高或比例。尺寸以实际发送给服务端的宽高为准。

V4 / V4.5 请求由适配器补齐以下原生提示词结构，用户只需填写普通提示词和负向提示词：

```json
{
  "v4_prompt": {
    "caption": {"base_caption": "<PROMPT>", "char_captions": []},
    "use_coords": false,
    "use_order": true
  },
  "v4_negative_prompt": {
    "caption": {"base_caption": "<NEGATIVE_PROMPT>", "char_captions": []},
    "legacy_uc": false
  }
}
```

## 超时、幂等与计费

服务端可能花费最长 `300` 秒获取完整结果，因此供应商默认 HTTP 超时为 `330` 秒。“最大尝试次数覆盖”默认 `1` 表示只发送一次请求，不是失败后再重试一次；填 `0` 才使用插件“运行控制”中的设置。

每个子请求会携带独立的 `Idempotency-Key`。同一子请求的重试固定使用同一个 API Key、种子、请求体和幂等键，避免网络故障后换 Key 或换参数形成新的付费生成请求。其他子请求使用不同的幂等键。

只对传输异常重试；HTTP 错误（包括 `409`、`429` 和 `5xx`）以及 `200` 响应中的无效图片结果都不会重新提交生成请求。

服务端不会重放之前的生成响应。同键同体请求会返回 `409 DUPLICATE_GENERATION_REQUEST:<job_id>:<status>`；同键不同请求体返回 `409 IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_PAYLOAD`。适配器遇到 `409` 会停止重试，不换新幂等键补发。

客户端超时、断开连接或插件取消任务，不代表服务端取消生成或退款。出现这些情况或重复请求错误时，请先在站点控制台生成日志核对原任务，再决定是否创建新任务。图生图会按服务端规则额外计费；多图与其他参数是否收费、令牌额度和免费渲染限制也以站点为准。

## 响应格式

适配器解析以下 JSON 图片结果，无需再从远端下载 ZIP：

```json
{
  "job": {
    "job_id": "example-job-id",
    "status": "succeeded"
  },
  "images": ["<RAW_IMAGE_BASE64>"],
  "image_format": "png"
}
```

返回的图片随后进入插件统一的审核、保存与发送流程。填写 NovelAI 官方 ZIP 接口或仅支持 OpenAI Images 请求结构的地址无法替代此 JSON 接口。

## English setup

Add **NovelAI native JSON API**, keep the default base URL `https://nai.rinko.ai`, enter the service token, and select `NovelAI/nai-diffusion-4-5-full` or another model available to your account. The adapter calls `POST /v1/nai/generate-image` with a native NovelAI payload and reads Base64 images from a synchronous JSON response.

Text-to-image requests use `action: "generate"`. A single reference image selects `action: "img2img"` and is sent as raw Base64 in `parameters.image`; strength and noise default to `0.7` and `0.0`. The reference is corrected for EXIF orientation, converted to RGB PNG, and resized proportionally with white padding to the requested dimensions. Multiple references are rejected. V4 / V4.5 prompt structures are assembled automatically. Inpainting, multiple characters, character references, Vibe, ZIP and streaming protocols are outside the current scope.

The default dimensions are `832×1216`. Each dimension must be a multiple of `64` within `64–2048`, and the total area must not exceed `3,145,728` pixels. An explicit aspect ratio recalculates dimensions from the configured pixel area and rounds each dimension to a multiple of `64`; out-of-range results are rejected before generation. The global resolution setting is ignored.

Keep the recommended timeout of `330` seconds and maximum attempt count of `1`. Only transport errors are retried, with the same token, seed, payload and idempotency key. HTTP errors and invalid image results are not resubmitted. A `409` stops retries because the service does not replay a previous response. Check the server generation log after a timeout, cancellation or duplicate-request error before submitting a new task. Image-to-image requests and other paid options follow the service’s pricing rules.
