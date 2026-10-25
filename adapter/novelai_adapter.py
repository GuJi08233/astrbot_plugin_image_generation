"""NovelAI native-payload adapter for YesNovelAI's JSON image endpoint."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import secrets
import time
from decimal import Decimal, InvalidOperation
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp
from PIL import Image, ImageOps

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION
from ..core.shared.logging import safe_log_error_body
from ..core.shared.types import GenerationRequest, GenerationResult, ImageCapability

RESPONSE_ERROR = "NovelAI response invalid; generation may have completed"
TRANSPORT_ERROR = "NovelAI transport error"


class NovelAIAdapter(BaseImageAdapter):
    """Generate and edit images using NovelAI parameters and JSON results."""

    DEFAULT_BASE_URL = "https://nai.rinko.ai"
    DEFAULT_MODEL = "nai-diffusion-4-5-full"
    ENDPOINT = "/v1/nai/generate-image"
    QUOTE_ENDPOINT = "/ai/generate-image/quote"
    STANDARD_SIZES = ((512, 512), (832, 1216), (1216, 832), (1024, 1024))

    def get_capabilities(self) -> ImageCapability:
        """Return supported capabilities selected in the provider configuration.

        Returns:
            Configured text, reference-image, and aspect-ratio capabilities.
        """
        return self._get_configured_capabilities() & (
            ImageCapability.TEXT_TO_IMAGE
            | ImageCapability.IMAGE_TO_IMAGE
            | ImageCapability.ASPECT_RATIO
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Check the quoted cost before generating with stable retry parameters.

        Args:
            request: Prompt, optional source image, and task metadata.

        Returns:
            A validated image or a safe error message. Quotes above the configured
            limit or unavailable quotes never proceed to generation. Only
            generation transport failures can retry with the original parameters.
        """
        if not self.api_keys:
            return GenerationResult(error="未配置 API Key")
        try:
            generation_url = self._endpoint_url()
            raw_limit = self.config.extra.get("max_cost_gems", 5)
            max_cost_gems = Decimal(str(raw_limit))
            if (
                isinstance(raw_limit, bool)
                or not max_cost_gems.is_finite()
                or max_cost_gems < 0
            ):
                raise ValueError("max_cost_gems must be a finite number >= 0")
            payload = await asyncio.to_thread(self._build_payload, request)
        except InvalidOperation:
            return GenerationResult(
                error="NovelAI 参数错误: max_cost_gems must be a finite number >= 0"
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return GenerationResult(error=f"NovelAI 参数错误: {exc}")

        api_key = self._get_current_api_key()
        quote_error = await self._check_cost_limit(
            request,
            url=generation_url.removesuffix(self.ENDPOINT) + self.QUOTE_ENDPOINT,
            payload=payload,
            api_key=api_key,
            max_cost_gems=max_cost_gems,
        )
        if quote_error:
            return GenerationResult(error=quote_error)
        idempotency_key = uuid4().hex
        error = "NovelAI 请求失败"
        for attempt in range(self.max_retry_attempts):
            if request.retry_status_callback:
                request.retry_status_callback(attempt + 1, self.max_retry_attempts)
            images, error = await self._generate_once(
                request,
                payload=payload,
                api_key=api_key,
                idempotency_key=idempotency_key,
            )
            if images is not None:
                return GenerationResult(images=images)
            error = error or "NovelAI 请求失败"
            if not self._is_retryable_error(error):
                break
            if attempt + 1 < self.max_retry_attempts:
                await asyncio.sleep(min(2**attempt, 10))
        return GenerationResult(error=error)

    async def _check_cost_limit(
        self,
        request: GenerationRequest,
        *,
        url: str,
        payload: dict[str, Any],
        api_key: str,
        max_cost_gems: Decimal,
    ) -> str | None:
        """Require a valid YesNAI quote within the provider's per-request limit.

        Args:
            request: Original image generation request for diagnostic context.
            url: Non-billing quote endpoint on the configured service.
            payload: Exact generation body, including the resolved random seed.
            api_key: Credential snapshot also used for generation and retries.
            max_cost_gems: Maximum quoted Gems allowed for this sub-request.

        Returns:
            An error that prevents generation, or None when the quote is allowed.
        """
        started = time.monotonic()
        try:
            async with self._get_session().post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                if response.status != 200:
                    return (
                        f"NovelAI 报价失败 (HTTP {response.status})，未提交生成；"
                        "请检查服务地址、令牌和报价接口"
                    )
                data = json.loads(await response.read(), parse_float=Decimal)
            if not isinstance(data, dict) or data.get("error"):
                return "NovelAI 报价响应无效，无法确认费用，未提交生成"
            total = data.get("total_gems")
            if isinstance(total, bool) or not isinstance(total, (int, Decimal)):
                return "NovelAI 报价缺少有效的 total_gems，未提交生成"
            quoted_cost = Decimal(total)
            if not quoted_cost.is_finite() or quoted_cost < 0:
                return "NovelAI 报价 total_gems 必须是有限非负数值，未提交生成"
            if quoted_cost > max_cost_gems:
                parameters = payload["parameters"]
                return (
                    f"NovelAI 报价 {quoted_cost} Gems 超过单次请求上限 "
                    f"{max_cost_gems} Gems，未提交生成"
                    f"（{parameters['width']}×{parameters['height']}，"
                    f"{parameters['steps']} 步）"
                )
            return None
        except (ValueError, UnicodeDecodeError, InvalidOperation):
            return "NovelAI 报价响应无效，无法确认费用，未提交生成"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._log_request_exception(request, time.monotonic() - started, exc)
            return "NovelAI 报价请求失败，无法确认费用，未提交生成"

    async def _generate_once(
        self,
        request: GenerationRequest,
        *,
        payload: dict[str, Any] | None = None,
        api_key: str | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """Send one native-payload request to the JSON generation endpoint.

        Args:
            request: Original image generation request.
            payload: Prepared payload shared by all retries of this request.
            api_key: Credential snapshot shared by all retries.
            idempotency_key: Unique key shared by all retries of this request.

        Returns:
            Validated image bytes or a safe error message.
        """
        payload = payload if payload is not None else self._build_payload(request)
        url = self._endpoint_url()
        headers = {
            "Authorization": f"Bearer {api_key or self._get_current_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": idempotency_key or uuid4().hex,
        }
        self._log_request_overview(request, url, payload=payload)
        self._log_debug_json("Request", payload, request.task_id)
        started = time.monotonic()
        try:
            async with self._get_session().post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                duration = time.monotonic() - started
                self._log_response_status(request, response.status, duration)
                if response.status != 200:
                    try:
                        body = await response.read()
                        error_text = body.decode("utf-8", errors="replace")
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        error_text = "The HTTP error response body could not be read"
                    self._log_api_error(request, response.status, duration, error_text)
                    return None, self._format_api_error_message(
                        response.status, error_text
                    )
                body = await response.read()
                try:
                    data = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    return None, f"{RESPONSE_ERROR}: expected a JSON response"
                self._log_debug_json("Response", data, request.task_id)
                return await asyncio.to_thread(self._extract_images, data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._log_request_exception(request, time.monotonic() - started, exc)
            detail = safe_log_error_body(exc) or type(exc).__name__
            return None, f"{TRANSPORT_ERROR}: {detail}"
        except Exception as exc:  # noqa: BLE001
            self._log_request_exception(request, time.monotonic() - started, exc)
            return None, f"{RESPONSE_ERROR}: {type(exc).__name__}"

    def _endpoint_url(self) -> str:
        """Build the supported native-payload JSON endpoint URL.

        Returns:
            A URL ending in /v1/nai/generate-image.

        Raises:
            ValueError: The base URL is invalid or points to a ZIP endpoint.
        """
        base = (self.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        parsed = urlsplit(base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Use an HTTP(S) service base URL without credentials")
        if "/native" in parsed.path or parsed.path.endswith("/ai/generate-image"):
            raise ValueError("Use the service root or /v1, not a ZIP endpoint")
        if base.endswith(self.ENDPOINT):
            return base
        if base.endswith("/v1"):
            return f"{base}/nai/generate-image"
        return f"{base}{self.ENDPOINT}"

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Build and validate one NovelAI native generation payload.

        Args:
            request: Prompt, source image, and optional target aspect ratio.

        Returns:
            Native parameters with standard dimensions unless a manual pixel size
            is supplied, a fixed seed, and a PNG source image when editing.

        Raises:
            ValueError: A parameter or source image is unsupported or invalid.
        """
        if len(request.images) > 1:
            raise ValueError("NovelAI img2img supports exactly one reference image")
        model = (self.model or self.DEFAULT_MODEL).strip()
        if request.images and model.startswith("nai-diffusion-5"):
            raise ValueError("NAI Diffusion 5 currently supports text-to-image only")
        if request.image_size is not None and (
            not isinstance(request.image_size, (tuple, list))
            or len(request.image_size) != 2
            or any(
                isinstance(side, bool) or not isinstance(side, int)
                for side in request.image_size
            )
        ):
            raise ValueError("Manual image size must contain two integer dimensions")

        parameters: dict[str, Any] = {}
        integer_fields = (
            ("width", 832, 64, 2048),
            ("height", 1216, 64, 2048),
            ("steps", 28, 1, 50),
            ("seed", -1, -1, 2**32 - 1),
        )
        for key, default, minimum, maximum in integer_fields:
            if request.image_size is not None and key in {"width", "height"}:
                raw = request.image_size[0 if key == "width" else 1]
            else:
                raw = self.config.extra.get(key, default)
            value = float(raw)
            if (
                isinstance(raw, bool)
                or not math.isfinite(value)
                or not value.is_integer()
            ):
                raise ValueError(f"{key} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
            parameters[key] = int(value)
        for key, default, maximum in (
            ("scale", 5.0, 10),
            ("cfg_rescale", 0.0, 1),
        ):
            raw = self.config.extra.get(key, default)
            value = float(raw)
            if (
                isinstance(raw, bool)
                or not math.isfinite(value)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{key} must be between 0 and {maximum}")
            parameters[key] = value

        width, height = parameters["width"], parameters["height"]
        if width % 64 or height % 64 or width * height > 3_145_728:
            raise ValueError("Dimensions must be multiples of 64 with area <= 3145728")
        if request.image_size is None:
            ratio = request.aspect_ratio
            target_ratio = width / height
            if ratio and ratio != UNSPECIFIED_OPTION:
                horizontal, vertical = (float(part) for part in ratio.split(":"))
                if not all(
                    math.isfinite(part) and part > 0 for part in (horizontal, vertical)
                ):
                    raise ValueError("Aspect ratio must contain two positive numbers")
                target_ratio = horizontal / vertical
            area = width * height
            # Match relative shape first; choose the nearest configured area for
            # the two square options. Automatic requests never invent a size.
            width, height = min(
                self.STANDARD_SIZES,
                key=lambda size: (
                    abs(math.log((size[0] / size[1]) / target_ratio)),
                    abs(size[0] * size[1] - area),
                ),
            )
        parameters.update(
            width=width,
            height=height,
            seed=secrets.randbits(32)
            if parameters["seed"] == -1
            else parameters["seed"],
            n_samples=1,
            sampler=str(self.config.extra.get("sampler", "k_euler_ancestral")),
            noise_schedule=str(self.config.extra.get("noise_schedule", "karras")),
            negative_prompt=str(self.config.extra.get("negative_prompt", "")),
            params_version=3,
        )
        if model.startswith(("nai-diffusion-4", "nai-diffusion-5")):
            parameters.update(
                legacy_v3_extend=False,
                legacy_uc=False,
                use_coords=False,
                v4_prompt={
                    "caption": {"base_caption": request.prompt, "char_captions": []},
                    "use_coords": False,
                    "use_order": True,
                },
                v4_negative_prompt={
                    "caption": {
                        "base_caption": parameters["negative_prompt"],
                        "char_captions": [],
                    },
                    "legacy_uc": False,
                },
            )
        if request.images:
            for config_key, key, default in (
                ("img2img_strength", "strength", 0.7),
                ("img2img_noise", "noise", 0.0),
            ):
                raw = self.config.extra.get(config_key, default)
                value = float(raw)
                if (
                    isinstance(raw, bool)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError(f"{config_key} must be between 0 and 1")
                parameters[key] = value
            try:
                with Image.open(BytesIO(request.images[0].data)) as source:
                    source = ImageOps.exif_transpose(source).convert("RGB")
                    source = ImageOps.pad(
                        source, (width, height), Image.Resampling.LANCZOS, color="white"
                    )
                    output = BytesIO()
                    source.save(output, format="PNG")
                    parameters["image"] = base64.b64encode(output.getvalue()).decode(
                        "ascii"
                    )
            except (
                OSError,
                ValueError,
                SyntaxError,
                Image.DecompressionBombError,
            ) as exc:
                raise ValueError("Reference image cannot be decoded") from exc
        return {
            "input": request.prompt,
            "model": model,
            "action": "img2img" if request.images else "generate",
            "parameters": parameters,
        }

    def _extract_images(self, data: Any) -> tuple[list[bytes] | None, str | None]:
        """Validate a complete JSON generation result before returning images.

        Args:
            data: Parsed JSON generation response.

        Returns:
            One verified image or a terminal response error.
        """
        if not isinstance(data, dict) or data.get("error"):
            return None, f"{RESPONSE_ERROR}: generation returned an error"
        job = data.get("job")
        status = job.get("status") if isinstance(job, dict) else None
        if isinstance(status, str):
            status = status.strip().lower()
        # YesNovelAI reports successful, settled generations as "charged".
        if status not in (None, "succeeded", "charged"):
            return None, f"{RESPONSE_ERROR}: job did not succeed"
        encoded_images = data.get("images")
        if not isinstance(encoded_images, list) or len(encoded_images) != 1:
            return None, f"{RESPONSE_ERROR}: expected exactly one final image"
        encoded = encoded_images[0]
        if not isinstance(encoded, str) or not encoded.strip():
            return None, f"{RESPONSE_ERROR}: missing image data"
        try:
            image = base64.b64decode(encoded, validate=True)
            with Image.open(BytesIO(image)) as decoded:
                decoded.verify()
            with Image.open(BytesIO(image)) as decoded:
                decoded.load()
        except (
            binascii.Error,
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ):
            return None, f"{RESPONSE_ERROR}: invalid Base64 or image data"
        return [image], None

    def _is_retryable_error(self, error: str) -> bool:
        """Retry only ambiguous transport failures with the original request.

        Args:
            error: Safe error message from a generation attempt.

        Returns:
            Whether another attempt with the same idempotency key is allowed.
        """
        return error.startswith(TRANSPORT_ERROR) and super()._is_retryable_error(error)
