"""
ComfyUI API client.
"""

import io
import time
import requests
from PIL import Image

from lib.comfyui.workflow import inject_image, get_workflow


class ComfyClient:
    """Client for interacting with ComfyUI API."""

    def __init__(self, url: str, api_token: str):
        self.url = url.rstrip("/") if url else url
        self.api_token = api_token

    def get_headers(self) -> dict:
        return {"ComfyUI-Api-Key": self.api_token}

    def upload_image(self, img: Image.Image) -> str:
        """Upload an image to ComfyUI."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"image": ("image.png", buf, "image/png")}
        response = requests.post(
            f"{self.url}/api/upload/image",
            files=files,
            headers=self.get_headers(),
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("name", data.get("filename", ""))

    def run_workflow(
        self,
        image_filename: str,
        target_megapixels: float = None,
        input_width: int = None,
        input_height: int = None,
    ) -> str:
        """Submit a workflow to ComfyUI for execution."""
        workflow = get_workflow()
        workflow_with_image = inject_image(
            workflow,
            image_filename,
            target_megapixels=target_megapixels,
            input_width=input_width,
            input_height=input_height,
        )

        response = requests.post(
            f"{self.url}/api/prompt",
            json={"prompt": workflow_with_image},
            headers=self.get_headers(),
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("prompt_id", "")

    def wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> dict:
        """Poll for workflow completion."""
        if not prompt_id:
            raise ValueError("No prompt_id returned from workflow execution")

        start_time = time.time()
        last_status_code = None

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Workflow {prompt_id} did not complete within {timeout}s"
                )

            response = requests.get(
                f"{self.url}/api/history/{prompt_id}",
                headers=self.get_headers(),
                timeout=30,
            )
            last_status_code = response.status_code

            if response.status_code == 200:
                data = response.json()
                if prompt_id in data:
                    prompt_data = data[prompt_id]
                    status = prompt_data.get("status", {})
                    if status.get("completed", False):
                        return prompt_data
                    if status.get("errored", False):
                        raise RuntimeError(
                            f"Workflow failed: {status.get('message', 'Unknown error')}"
                        )
                    extra = prompt_data.get("executed", {})
                    if extra:
                        return prompt_data

            time.sleep(poll_interval)

    def download_output(self, filename: str) -> Image.Image:
        """Download an output image from ComfyUI."""
        response = requests.get(
            f"{self.url}/api/view",
            params={"filename": filename},
            headers=self.get_headers(),
            timeout=60,
        )
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
