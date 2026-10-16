# Codex Responses 接口配置

`codex_responses` 适配器用于接入采用 OpenAI Responses 风格的图片生成接口。它是专用模板：插件固定构造流式请求，通过 `tool_choice` 指定调用生图工具、发送 Bearer API Key，并识别 `image_generation_call` 中的最终图片结果，不需要手动配置请求体或图片路径。

当前版本支持：

- `POST` 请求，默认发送 `stream: true`；接口路径默认为 `/codex/responses`，可改为 `/v1/responses` 等
- `Authorization: Bearer <API Key>` 认证
- 根据响应 `Content-Type` 增量解析 SSE，兼容普通 JSON 和直接图片响应
- 文生图和参考图驱动的图像编辑
- 标准 `output[*].type == "image_generation_call"` 的 `result` 图片字段
- 纯 Base64、`data:image/...;base64,...` 和 HTTP(S) 图片 URL
- 兼容 `result_b64`、`b64_json`、`base64` 以及 `data[*]` 包装的图片结果
- API Key 轮换、代理、超时、重试与现有任务发送流程

暂不支持：

- 宽高比与分辨率参数
- 将流式中间预览逐步发送到聊天
- 返回任务 ID 后需要轮询的异步接口
- 下载图片 URL 时额外转发 API Key 或自定义下载请求头

## 快速配置

1. 在“图像模型供应商”中新增 **Codex Responses 接口**。
2. 填写供应商名称，例如 `宝宝AI`。
3. 在“接口地址”填写服务根地址，例如 `https://baobao-ai.com`。
4. 在“接口路径”填写实际路径；默认 `/codex/responses`。标准 OpenAI Responses 填 `/v1/responses`。
5. 在“API 密钥”填写宝宝 AI API Key。
6. 在“可用模型列表”填写服务实际支持的模型，例如 `gpt-5.6-terra`。
7. 在“生图模型”中选择 `宝宝AI/gpt-5.6-terra`。

实际请求地址由接口地址和接口路径直接拼接，不再猜测或改写路径：

| 接口地址 | 接口路径 | 实际请求地址 |
| :--- | :--- | :--- |
| `https://baobao-ai.com` | `/codex/responses` | `https://baobao-ai.com/codex/responses` |
| `https://api.example.com` | `/v1/responses` | `https://api.example.com/v1/responses` |
| `https://chatgpt.com/backend-api` | `/codex/responses` | `https://chatgpt.com/backend-api/codex/responses` |

接口地址必须是 `http://` 或 `https://` URL，且不能包含查询参数或片段。不要把完整接口路径填进接口地址。

## 固定请求格式

每张图片都会发起一个独立请求；插件现有的多图任务会按请求级并发配置调度这些请求。

请求头：

```http
Authorization: Bearer <API Key>
Content-Type: application/json
Accept: text/event-stream, application/json, image/*
```

请求体固定为：

```json
{
  "model": "gpt-5.6-terra",
  "stream": true,
  "input": "生成一张图片：白色背景中央有一个小的纯蓝色实心圆，不要文字。",
  "tool_choice": {"type": "image_generation"},
  "tools": [
    {
      "type": "image_generation",
      "output_format": "png"
    }
  ]
}
```

其中 `input` 使用插件经过预设、人设和附加提示词合并后的最终文本。文本生图不会额外添加 `messages`、`prompt`、`n`、`modalities`、`size`、参考图或尺寸字段。工具声明会固定带 `output_format: "png"`。文生图和图生图均发送 `stream: true`；服务若仍返回 JSON 或图片，插件会按实际响应类型解析，不会重新提交非流式请求来降级。

文生图和图生图均发送 `tool_choice: {"type":"image_generation"}`，明确要求模型调用生图工具。这样 `/生图 原神` 等仅含主题的提示词也会按生图任务处理，无需用户额外添加“画一个”。提示词正文保持原样。接入服务需要支持这一 Responses 字段；第三方中转是否接受该字段仍需在实际服务上验证。

## 图像编辑请求

当用户发送、引用或通过 LLM 工具/公共 API 提供参考图时，插件会复用既有的参考图收集、大小校验、文件类型校验、去重和转换链路，并把每张 `ImageData` 按原顺序编码为 `data:<mime>;base64,...`。此时请求使用 Responses 多模态 `input`：

```json
{
  "model": "gpt-5.6-terra",
  "stream": true,
  "tool_choice": {"type": "image_generation"},
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "将这张图片改成赛博朋克夜景风格，保留人物姿势和画面构图。"
        },
        {
          "type": "input_image",
          "image_url": "data:image/jpeg;base64,<YOUR_BASE64_IMAGE>"
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "image_generation",
      "output_format": "png"
    }
  ]
}
```

多模态 `input` 格式与宝宝 AI 的 `gpt-5.6-terra` 编辑请求和同步成功响应已经核验一致；`tool_choice` 和流式请求尚未在该服务上实测。最终响应中的输出仍从：

```text
response.output[i].type == image_generation_call
response.output[i].result
```

读取，`result` 是 Base64 编码的图片二进制。

- 直接发送或引用图片后，可使用 `/生图 <编辑要求>`。
- LLM 工具的 `reference_images`、`avatar_references` 和带图片的人设会进入同一条图生图链路。
- 插件公共 API 的 `reference_image_sources` 和 `reference_image_data` 也会进入同一条图生图链路。
- 多张参考图会全部按顺序作为 `input_image` 发送；宝宝 AI 已验证单张输入。若当前账号或模型限制多图，请只提供一张参考图，插件不会静默截断。

宽高比和分辨率当前仍不会加入 Codex 请求。

## 响应格式

当响应 `Content-Type` 为 `text/event-stream` 时，插件会增量读取 SSE 事件，解析 `response.completed` 中的最终响应，也支持从 `response.output_item.done` 的 `image_generation_call` 读取最终图片结果。例如：

```text
event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"image_generation_call","status":"completed","result":"<Base64 编码的 PNG 图像数据>"}}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"image_generation_call","status":"completed","result":"<Base64 编码的 PNG 图像数据>"}]}}

```

`response.image_generation_call.partial_image` 等中间预览不会作为最终结果或发送到聊天。最终图片仍统一经过原有的审核、保存和发送流程。

普通 JSON 响应保持兼容，标准 Responses 响应形态如下：

```json
{
  "id": "resp_example",
  "status": "completed",
  "error": null,
  "output": [
    {
      "type": "reasoning"
    },
    {
      "type": "image_generation_call",
      "status": "generating",
      "revised_prompt": "...",
      "result": "<Base64 编码的 PNG 图像数据>"
    },
    {
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "..."
        }
      ]
    }
  ]
}
```

插件会扫描所有 `output` 项，筛选 `type` 为 `image_generation_call` 的项目，再优先读取其 `result`。即使该调用项的 `status` 仍是 `generating`，只要同一响应已经给出可解码的 `result`，插件仍会将其作为图片结果处理。

`result` 可以是：

- 纯 Base64 图像数据（宝宝 AI 实测格式）
- `data:image/png;base64,...` 形式的 data URL
- 可公开下载的 `http://` 或 `https://` 图片 URL

为兼容常见中转实现，插件也会尝试 `result_b64`、`b64_json`、`base64`、`url`、顶层等价字段以及 `data[*]` 下的图片字段。

如果接口只返回任务 ID、`202 Accepted`，或者图片 URL 下载需要另一种认证方式，则此专用适配器不会完成图片生成；异步任务需要独立的轮询适配器。流式请求、事件格式及工具选择字段是否被第三方中转接受，仍需在实际服务上验证。

## 能力边界

当前 provider 支持“文生图”和“图生图”。当有参考图时，插件会使用上文的多模态 `input_image` data URL；当没有参考图时，插件保留已验证的字符串 `input` 请求格式。

以下参数仍由统一任务执行器忽略，不会进入 Codex 请求：

- 宽高比
- 分辨率

请勿在服务端猜测或复用其他 OpenAI 接口的 `size` 或 `quality` 字段；待 Codex Responses 提供明确协议后再扩展。

## 排障

### 出现 `Server disconnected`，但任务最终成功

这不是本插件把请求超时缩短到约 150 秒。Codex provider 的“超时时间覆盖”会传给 `aiohttp.ClientTimeout(total=...)`；例如配置为 `300` 时，客户端总请求超时仍是 300 秒，流式事件不会重置总超时。`Server disconnected` 表示服务端、显式代理、透明代理、负载均衡、CDN 或其他网络设备主动关闭了底层连接。

插件会将这种无 HTTP 状态码的连接错误视为可重试错误。因此日志可能先显示一次连接异常，随后由重试请求成功，任务最终仍正确保存并发送图片。新版日志会在“适配器请求”与“请求异常”中标出配置的超时秒数，便于确认实际生效值。

排查与解决顺序：

1. 在 Codex provider 中暂时清空“代理地址”后复测；如果问题消失，调整该代理的 HTTP read/idle timeout。
2. 在宝宝 AI、反向代理、CDN/WAF 或负载均衡侧，将上游/下游的 read、response 或 idle timeout 设为高于插件请求超时（300 秒应留出缓冲）。
3. 从运行 AstrBot 的同一台机器直接向服务发送同样请求，比较是否仍在约 150 秒断开，以区分插件外的网络链路。
4. 当前适配器默认请求 SSE；确认服务端实际返回 `Content-Type: text/event-stream`，并确认中间代理没有缓冲事件。只有服务端持续发送且代理及时转发数据时，流式响应才可能改善空闲断连；它不会绕过总超时，也不保证解决所有断连。仅提供异步任务 ID 的服务仍需其他适配器。

不要盲目增加客户端 timeout 或无限重试：远端可能已在第一次连接断开前完成生成，重复提交会带来重复生成或重复计费风险。

### `WebSocket API call timeout`

若日志模块名不是 `astrbot_plugin_image_generation`，该错误不由本插件的 Codex HTTP 请求产生。例如 `[astrbot_plugin_videos_analysis.main:...]` 应在视频分析插件或 AstrBot 的 WebSocket 发送链路中单独排查。

### 返回 `API 错误 (401)` 或 `API 错误 (403)`

确认 API Key 是否正确、是否仍有权限，并确保服务端采用标准 Bearer 认证。

### 返回“未配置 Codex Responses 接口地址”

在 provider 中填写服务根地址。不要只填写路径，也不要填写带 query 的临时链接。

### 返回“响应中未找到图片数据”

确认服务端在同一次 JSON 响应或 SSE 的 `response.completed.response` 中返回：

```text
output[*].type == image_generation_call
output[*].result
```

SSE 也可通过 `response.output_item.done.item` 返回带 `result` 的最终 `image_generation_call`。只有 `partial_image`、文本增量或任务 ID 不算最终图片结果；同时检查响应头是否正确标为 `text/event-stream`。

若服务端字段不同，可先保留一段脱敏响应并据此扩展专用适配器；也可以改用 `custom_http` 手工配置请求体和图片提取路径。

### 生成图片 URL 下载失败

优先让服务端返回 `result` 的 Base64 或 data URL。当前适配器为避免把 API Key 发送到未知图片域名，不会在下载图片 URL 时携带生成接口的 Authorization 头。

### 接口拒绝流式请求或只返回异步任务

当前 provider 固定发送 `stream: true`，不提供流式开关，也不会在请求失败后自动重发 `stream: false`，以免重复生成或计费。服务可以在同一次请求中返回 SSE 最终结果、普通 JSON 或图片；若服务拒绝流式参数，需要服务端支持该请求或改用可自定义请求体的适配器。只返回任务 ID、需要后续轮询的协议仍不支持。
