import re
import json
import base64
import uuid
import requests
import numpy as np
import copy
from typing import Optional, List, Dict, Any
from PIL import Image
from io import BytesIO
from math import ceil, floor

from verl.workers.agent.tool_envs import ToolBase

INITIALIZATION_CODE_TEMPLATE = """
from PIL import Image
import base64
from io import BytesIO

_img_base64 = "{base64_image}"
image = Image.open(BytesIO(base64.b64decode(_img_base64)))
"""

CODE_EXECUTION_TEMPLATE = """Code execution result:
stdout:
```
{stdout}
```

stderr:
```
{stderr}
```

{image}
"""

def pil_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def base64_to_pil_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


class CodeToolBoxV2(ToolBase):
    name = "code_toolbox_v2"
    code_sandbox_url = "http://127.0.0.1:12345/jupyter_sandbox"
    session_id = str(uuid.uuid4())
    max_images_per_round = 3

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed

    def extract_answer(self, action_string: str) -> str:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None

    def extract_python_code(self, action_string: str) -> str:
        tool_call_match = re.findall(r'<code>(.*?)</code>', action_string, re.DOTALL)
        if not tool_call_match:
            return None
        
        last_code_block = tool_call_match[-1]
        pattern = r'```python\s*\n(.*?)\n```'
        code_match = re.findall(pattern, last_code_block, re.DOTALL)
        if not code_match:
            return None
        return code_match[-1]

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}

        code_string = self.extract_python_code(action_string)
        if not code_string:
            return "", 0.0, True, {}

        exec_ret = self.request_jupyter_execution(code_string)
        if not exec_ret or exec_ret['status'] != 'success':
            obs = "Code execution error"
            return obs, 0.0, True, {"error": "Code execution failed"}

        image_list = exec_ret.get('images', [])
        image_list = image_list[:self.max_images_per_round]
        code_result_string = CODE_EXECUTION_TEMPLATE.format(
            stdout=exec_ret.get('stdout', ''),
            stderr=exec_ret.get('stderr', ''),
            image="Images:\n" + "<image>" * len(image_list) if len(image_list) > 0 else "",
        ).strip()

        if len(image_list) == 0:
            obs = "<|im_end|>\n<|im_start|>user\n" + code_result_string + "<|im_end|>\n<|im_start|>assistant\n<think>"
            print(f' [DEBUG code] Code success without images: {action_string=}')
            return obs, 0.0, False, exec_ret
        else:
            obs = {
                "prompt": "<|im_end|>\n<|im_start|>user\n" + code_result_string + "<|im_end|>\n<|im_start|>assistant\n<think>",
                "multi_modal_data": {"image": image_list},
            }
            print(f' [DEBUG code] Code success with images: {action_string=}')
            return obs, 0.0, False, exec_ret

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

        base64_image = pil_image_to_base64(self.multi_modal_data['image'][0])
        init_code_string = INITIALIZATION_CODE_TEMPLATE.format(base64_image=base64_image)
        init_ret = self.request_jupyter_execution(init_code_string)

        if not init_ret or init_ret['status'] != 'success':
            print(f' [ERROR code] Initialization code execution failed: {init_ret}')
        return init_ret

    def request_jupyter_execution(self, code_string, code_timeout=10, request_timeout=20):
        try:
            resjson = requests.post(
                self.code_sandbox_url,
                json={
                    "session_id": self.session_id,
                    "code": code_string,
                    "timeout": code_timeout
                },
                timeout=request_timeout
            ).json()
            result_dict = resjson['output']
        except Exception as err:
            print(f' [ERROR code] Request to Jupyter sandbox failed: {err}')
            return None

        image_pil_list = []
        image_base64_list = result_dict.get("images", [])
        for idx, img in enumerate(image_base64_list):
            try:
                img_pil = base64_to_pil_image(img)
                img_pil = self.maybe_resize_image(img_pil)
                image_pil_list.append(img_pil)
            except Exception as err:
                print(f' [ERROR code] Failed to decode image {idx}: {err}')
                continue

        return dict(
            status=resjson.get("status", "error"),
            execution_time=resjson.get("execution_time", -1.0),
            result=result_dict.get("result", ""),
            stdout=result_dict.get("stdout", ""),
            stderr=result_dict.get("stderr", ""),
            images=image_pil_list,
        )

    def maybe_resize_image(self, image):
        """
        Qwen-VL raises an error for images with height or width less than 32 pixels.
        """
        height, width = image.height, image.width
        if min(height, width) >= 32:
            return image

        ratio = 32 / min(height, width)
        new_height = ceil(height * ratio)
        new_width = ceil(width * ratio)
        new_image = image.resize((new_width, new_height), Image.LANCZOS)
        return new_image


if __name__ == "__main__":
    debug_action = """
The skateboard appears to be a dark color, possibly black or dark gray, with lighter-colored wheels. However, the exact shade and the distinction between the board and wheels makes it hard to determine the precise color without further detail. The skateboard's deck looks similar to what I see on the majority of boards in standard skate parks.

However, to confirm the color accurately, analyzing the pixel ranges of the skateboard's deck and wheels is required. We can use Python to process the image.

<code>
```python
from PIL import Image
import numpy as np

# Open the image
image = Image.open('path_to_image')

# Convert to array
img_array = np.array(image)

# Find the sands of skateboard (black or dark colors w/ yellow wheels)
skateboard_img = img_array[height_of_skateboard : , width_of_skateboard : ]

# check pixel values and analyze color distribution
skateboard_img_pixels = skateboard_img.reshape(-1,4)

# Considering the value range for黯 and yuellow wheels
black_range = np.logical_and(skateboard_img_pixels[:,3] < 50, np.logical_or(skateboard_img_pixels[:,0] < 30, skateboard_img_pixels[:,1] < 50))  # x, y, z, a for REGB
yellow_range = np.logical_or(skateboard_img_pixels[:,0] > 250, skateboard_img_pixels[:,1] > 250, skateboard_img_pixels[:,2] > 250, skateboard_img_pixels[:,3] < 30)

# If the pixel part covers black range and yellow range, label as skateboard
# This solution begins with rectangle slicing, images need ratios and antenna should be done 
```</code>
""".strip()

    # tool = CodeToolBoxV2("code_toolbox_v2", 2, 3)
    # tool.action()
