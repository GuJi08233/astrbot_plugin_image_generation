"""NovelAI native-payload adapter for YesNovelAI's JSON image endpoint."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import secrets
import time
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
        """Generate images while keeping credentials and payload stable on retry.

        Args:
            request: Prompt, optional source image, and task metadata.

        Returns:
            A validated image or a safe error message. Only transport failures
            can retry; the service does not replay previous generation results.
        """
        if not self.api_keys:
            return GenerationResult(error="未配置 API Key")
        try:
            self._endpoint_url()
            payload = await asyncio.to_thread(self._build_payload, request)
        except (TypeError, ValueError, OverflowError) as exc:
            return GenerationResult(error=f"NovelAI 参数错误: {exc}")

        api_key = self._get_current_api_key()
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
            Native parameters with a fixed seed and PNG source image when editing.

        Raises:
            ValueError: A parameter or source image is unsupported or invalid.
        """
        if len(request.images) > 1:
            raise ValueError("NovelAI img2img supports exactly one reference image")
        model = (self.model or self.DEFAULT_MODEL).strip()
        if request.images and model.startswith("nai-diffusion-5"):
            raise ValueError("NAI Diffusion 5 currently supports text-to-image only")

        parameters: dict[str, Any] = {}
        integer_fields = (
            ("width", 832, 64, 2048),
            ("height", 1216, 64, 2048),
            ("steps", 28, 1, 50),
            ("seed", -1, -1, 2**32 - 1),
        )
        for key, default, minimum, maximum in integer_fields:
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
        ratio = request.aspect_ratio
        if ratio and ratio != UNSPECIFIED_OPTION:
            horizontal, vertical = (float(part) for part in ratio.split(":"))
            if not all(
                math.isfinite(part) and part > 0 for part in (horizontal, vertical)
            ):
                raise ValueError("Aspect ratio must contain two positive numbers")
            area = width * height
            width = round(math.sqrt(area * horizontal / vertical) / 64) * 64
            height = round(math.sqrt(area * vertical / horizontal) / 64) * 64
            if not (64 <= width <= 2048 and 64 <= height <= 2048):
                raise ValueError("Aspect ratio produces dimensions outside 64..2048")
            if width * height > 3_145_728:
                raise ValueError(
                    "Aspect ratio produces an image larger than 3145728 pixels"
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
