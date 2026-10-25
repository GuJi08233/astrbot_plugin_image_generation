# NovelAI 原生 JSON 接口配置

`novelai` 适配器使用 NovelAI 原生请求结构调用 `POST /v1/nai/generate-image`，解析同步 JSON 响应中的 `images` Base64 图片。生成前先调用 `POST /ai/generate-image/quote` 检查费用。默认服务地址为 `https://nai.rinko.ai`，兼容服务需要同时支持报价接口和相同的生成请求、响应格式。

这个接口与 `/native/ai/generate-image` 使用相同的原生请求结构，但返回 JSON。图生图由 `action: "img2img"` 和 `parameters.image` 调用，不需要处理 ZIP 或 MessagePack。

当前支持：

- 文生图和单张参考图图生图。
- 自动组装 V4 / V4.5 的 `v4_prompt` 与 `v4_negative_prompt`。
- 聊天/LLM 四种标准尺寸匹配、WebUI 手动宽高，以及步数、采样器、CFG、种子、负向提示词、噪声调度和 CFG 重缩放。
- 图生图重绘强度和噪声。
- 插件统一的多图任务、队列、参考图校验、审核、额度和结果发送流程。
- Bearer Token、代理、请求超时、幂等键和供应商级单次请求 Gems 报价上限。

当前不支持局部重绘、多角色提示词、角色精确参考、Vibe、ZIP 返回值、MessagePack 或 SSE 流式生成。质量标签和 UC 预设不会自动追加；需要的标签应直接写入提示词或预设。图生图仅支持一张参考图，多张参考图会明确报错。

## 快速配置

1. 在“图像模型供应商”中新增 **NovelAI 原生 JSON 接口**。
2. 供应商名称可保留 `NovelAI`，接口地址保留 `https://nai.rinko.ai`。
3. 在“API 密钥”中填写该站点的令牌，例如 `ynai-…`。
4. 根据账号权限维护可用模型。默认提供 `nai-diffusion-4-5-full` 和 `nai-diffusion-4-5-curated`。
5. 在“生图模型”或 `/生图模型` 中选择 `NovelAI/nai-diffusion-4-5-full`。
6. 初次使用可保留默认宽高 `832×1216`、超时 `330` 秒和最大尝试次数 `1`。
7. 将“单次请求 Gems 上限”（`max_cost_gems`）设为可接受的费用，例如 `2`、`3` 或 `5`；默认为 `5`，填 `0` 仅允许零报价。现有供应商未配置此项时也按 `5` 处理。

默认启用“文生图”“图生图”和“宽高比”。聊天指令与 LLM 工具只生成四种标准尺寸，宽高比用于从中匹配尺寸；WebUI 选择 NovelAI 后，使用宽度、高度输入框代替通用宽高比和分辨率。全局“默认分辨率”及单次请求的 `1K` / `2K` / `4K` 不参与 NovelAI 尺寸计算。

## 文生图和图生图

没有参考图时，`/生图 1girl, masterpiece, best quality` 使用 `action: "generate"`。V4 系列模型建议使用模型支持的英文标签和提示词。

图生图可以直接发送一张图片并输入 `/生图 <修改要求>`，也可以引用一条包含一张图片的消息。插件会收集参考图并将它编码为不带 `data:` 前缀的 Base64，发送 `action: "img2img"`、`parameters.image`、`parameters.strength` 与 `parameters.noise`。

参考图来源仍包括当前消息、引用消息、@ 用户头像和人设图片。请确保去重后的参考图合计只有一张；例如消息图片与另一张人设参考图同时存在时，会超过此适配器的上限。不会静默丢弃其中一张图。

默认重绘强度为 `0.7`。减小该值通常更接近原图，增大通常使重绘变化更大；噪声默认 `0.0`。参考图通过插件既有的图片安全校验后，会校正 EXIF 方向、转换为 RGB PNG，并按目标宽高等比缩放，以白色填充不足的区域；发送的是处理后的图片，避免不同原图尺寸导致请求不匹配。

单个插件任务仍可以生成多张图片。每个子请求发送 `n_samples: 1`，拥有独立的幂等键；文档中服务端支持的 `n_samples: 1–8` 不作为额外批量配置开放。

## 参数与尺寸

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| 单次请求 Gems 上限 | `5` | 非负数；生成前检查报价，超限或报价失败不生成。每个子请求独立限制，`0` 仅允许零报价。 |
| 默认图片宽度 | `832` | 聊天/LLM 标准尺寸匹配的比例与面积基准，也是 WebUI 宽度默认值。 |
| 默认图片高度 | `1216` | 聊天/LLM 标准尺寸匹配的比例与面积基准，也是 WebUI 高度默认值。 |
| 采样步数 | `28` | 范围 `1–50`。 |
| 采样器 | `k_euler_ancestral` | 填写服务与模型支持的原生采样器名称。 |
| 提示词遵循强度（CFG） | `5.0` | 范围 `0–10`。 |
| 随机种子 | `-1` | 每个子请求随机选择种子；非负整数表示固定种子，重试保持不变。 |
| 负向提示词 | 空 | 自动用于普通负向提示词和 V4 负向结构。 |
| 噪声调度 | `karras` | 需与模型、采样器兼容。 |
| CFG 重缩放 | `0.0` | 范围 `0–1`。 |
| 图生图重绘强度 | `0.7` | 范围 `0–1`，仅图生图发送。 |
| 图生图噪声 | `0.0` | 范围 `0–1`，仅图生图发送。 |

聊天指令、LLM 工具和未指定手动尺寸的公共 API 请求只从以下四种尺寸选择：

- `512×512`
- `832×1216`
- `1216×832`
- `1024×1024`

适配器按请求的宽高比选择比例最接近的尺寸；未指定宽高比时使用供应商默认宽高的比例。多个候选比例相同时，再选择面积更接近供应商默认 `width×height` 的尺寸，因此可通过供应商默认宽高选择较小或较大的方图。使用默认 `832×1216` 时，`3:4` 匹配 `832×1216`，`4:3` 匹配 `1216×832`，`1:1` 匹配 `1024×1024`；默认宽高为 `512×512` 时，方图匹配 `512×512`。

WebUI 选择 NovelAI 后可手动填写宽度和高度，初始值取自供应商默认配置。该手动尺寸可超出四种标准尺寸，并覆盖宽高比、预设中的比例及供应商默认宽高，但必须满足：

- 每边 `64–2048` 像素，且是 `64` 的倍数。
- 总面积不超过 `3,145,728` 像素。

其他插件也可通过公共 API 的 `image_size=(width, height)` 显式指定 NovelAI 手动尺寸；不传或传 `None` 则使用四种标准尺寸匹配。该参数不向聊天指令或 LLM 工具暴露。详见[公共 API 文档](public-api.md)。手动尺寸无论是否在标准列表中，仍须通过相同的 Gems 报价检查；其他适配器维持原有尺寸行为。

修复前，默认 `832×1216` 面积下的 `3:4` 请求会换算为 `896×1152`，不在免费尺寸内；现在聊天/LLM 请求会匹配 `832×1216`。四种标准尺寸本身不保证免费：免费标准出图还要求单张文生图、最多 `28` 步，且不含图生图、局部重绘、参考或 SMEA 等付费附加功能；任一条件不满足都会按全价计费，不做部分减免。插件不自动降低步数，最终请求体仍先查询报价，超限或报价失败即停止生成。

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

文生图和图生图的每个子请求都会先携带同一个 API Key 和完整生成请求体调用 `POST /ai/generate-image/quote`。报价不扣费、不占用生成队列；插件读取 JSON 顶层的 `total_gems`，仅在报价为有效非负数且不超过供应商的 `max_cost_gems` 时提交生成。报价超限、请求失败、响应无效或缺少有效价格时都会停止，不会跳过报价直接生成。例如上限为 `5`、报价为 `80` Gems 时，会在生成前拒绝该子请求。

该上限按每个模型接口子请求独立计算，不是整个多图任务的总预算。例如任务包含 `3` 个子请求且上限为 `5` 时，通过报价检查的合计金额最多可为 `15` Gems。每个子请求仍发送 `n_samples: 1`。

报价是预估，不锁定价格或额度。服务端在生成时重新计价，并优先使用免费额度和图片包，因此客户端报价检查不等于服务端硬性扣费上限。需要严格限制实际扣费时，请同时在站点 API 令牌中设置单次请求 Gems 上限；服务端超限会返回 `TOKEN_PER_REQUEST_GEMS_LIMIT_EXCEEDED`。实时免费规格、计价和令牌设置以[站点文档](https://nai.rinko.ai/docs)及价格页面为准。

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
    "status": "charged"
  },
  "images": ["<RAW_IMAGE_BASE64>"],
  "image_format": "png"
}
```

YesNovelAI 的实际成功响应使用 `job.status: "charged"` 表示生成成功并已结算，适配器同时兼容文档中的 `"succeeded"`，忽略状态字符串的大小写和前后空白。返回状态时，只接受这两个成功状态；缺少状态时仍兼容直接返回图片的响应。图片必须能够完整解码，失败、处理中或未知状态不会因为包含图片而被接受。

返回的图片随后进入插件统一的审核、保存与发送流程。填写 NovelAI 官方 ZIP 接口或仅支持 OpenAI Images 请求结构的地址无法替代此 JSON 接口。

## English setup

Add **NovelAI native JSON API**, keep the default base URL `https://nai.rinko.ai`, enter the service token, and select `NovelAI/nai-diffusion-4-5-full` or another model available to your account. The adapter calls `POST /v1/nai/generate-image` with a native NovelAI payload and reads Base64 images from a synchronous JSON response.

Set **Gems limit per request** (`max_cost_gems`) to an acceptable amount, such as `2`, `3`, or `5`. The default is `5`, including existing providers without this field. The value must be nonnegative; `0` allows only zero-price quotes. Before each text-to-image or image-to-image sub-request, the adapter calls `POST /ai/generate-image/quote` with the same API key and complete generation payload. Quotes are free and do not enter the generation queue. A missing or invalid `total_gems`, failed quote request, or quote above the limit stops generation. Compatible services must support both the quote and generation endpoints.

The limit applies separately to each sub-request, not to the whole multi-image task. Three sub-requests with a limit of `5` can pass with quotes totaling up to `15` Gems. An `80`-Gems quote is rejected when the configured limit is `5`. The adapter quotes the final dimensions and sampling steps without changing them after the quote.

Quotes do not reserve a price or allowance. The server recalculates the price at generation time and applies free allowances and image packs first, so this client check is not a server-enforced spending cap. For a strict cap on the actual charge, also configure the API token's per-request Gems limit on the site; exceeding it returns `TOKEN_PER_REQUEST_GEMS_LIMIT_EXCEEDED`. Consult the [service documentation](https://nai.rinko.ai/docs) and live pricing page for current rules.

Text-to-image requests use `action: "generate"`. A single reference image selects `action: "img2img"` and is sent as raw Base64 in `parameters.image`; strength and noise default to `0.7` and `0.0`. The reference is corrected for EXIF orientation, converted to RGB PNG, and resized proportionally with white padding to the requested dimensions. Multiple references are rejected. V4 / V4.5 prompt structures are assembled automatically. Inpainting, multiple characters, character references, Vibe, ZIP and streaming protocols are outside the current scope.

Chat commands, LLM tools, and public API requests without an explicit manual size select only from `512×512`, `832×1216`, `1216×832`, and `1024×1024`. The closest aspect ratio wins; if no ratio is requested, the provider's default width/height ratio is used. Candidates with the same ratio are compared by their area against the provider's configured pixel area. With the default `832×1216`, `3:4` selects `832×1216`, `4:3` selects `1216×832`, and `1:1` selects `1024×1024`. Setting provider defaults to `512×512` selects the smaller square for square requests. The global resolution setting is ignored.

When NovelAI is selected in the WebUI, width and height inputs replace the generic aspect-ratio and resolution controls. They start from the provider defaults and allow manual dimensions outside the four standard sizes. A manual size overrides aspect ratios, including preset ratios, and provider dimensions. Each dimension must be a multiple of `64` within `64–2048`, and the total area must not exceed `3,145,728` pixels. Plugin callers can use `image_size=(width, height)` on the [public API](public-api.md); omitting it or passing `None` keeps standard-size matching. Chat commands and LLM tools do not expose manual dimensions. All manual requests still pass the same Gems quote check, and other adapters keep their existing size behavior.

Before this fix, a `3:4` request with the default area became `896×1152`, outside the free sizes; chat/LLM requests now select `832×1216`. A standard size alone does not guarantee a free render: free standard renders also require one text-to-image image, no more than `28` steps, and no paid add-ons such as image-to-image, inpainting, references, or SMEA. Missing any condition incurs the full price without partial waivers. Sampling steps are not reduced automatically; an over-limit or failed quote stops generation.

Keep the recommended timeout of `330` seconds and maximum attempt count of `1`. Only transport errors are retried, with the same token, seed, payload and idempotency key. HTTP errors and invalid image results are not resubmitted. A `409` stops retries because the service does not replay a previous response. Check the server generation log after a timeout, cancellation or duplicate-request error before submitting a new task. Image-to-image requests and other paid options follow the service’s pricing rules.

YesNovelAI returns `job.status: "charged"` for successful, settled generations. The adapter also accepts the documented `"succeeded"` state, ignoring case and surrounding whitespace, and still requires a fully decodable image. Failed, pending, or unknown states are rejected even if an image is present.
