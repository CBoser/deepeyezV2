import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from verl.workers.agent.envs.mm_process_engine.code_prompt import PROMPT
from math import ceil, floor
import random
import requests
from io import BytesIO
import time
import shutil
import os


class VisualToolBoxV6(ToolBase):
    name = "visual_toolbox_v6"
    user_prompt = PROMPT.USER_PROMPT_V1
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed

        self.root_tmp_path = '/diancpfs/user/honglingyi/TRAIN_TMP'
        self.query_urls = [
            "http://10.39.14.88:8000/query",
        ]


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        code_call_match = re.findall(r'<code>(.*?)</code>', action_string, re.DOTALL)
        return code_call_match[-1] if code_call_match else None

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
        action = self.extract_action(action_string)
        # print (f'[DEBUG] EXECUTE ACTION {action_string=}, {action=}')
        if not action:
            return "", 0.0, True, {}
            
        exec_str, add_image = self.exec_code(action)
        # print(f'[DEBUG] EXEC CODE {exec_str=}, {add_image=}')

        try:
            exec_str, add_image = self.exec_code(action)

            if not exec_str:
                obs = "\n<|im_start|>user\n" + f"Code Error. Not Valid Python Code." + "<|im_end|>\n<|im_start|>assistant\n"
                reward = 0.0 
                done = False
                info = {"error": 'No Code', "status": "failed"}
                return obs, reward, done, info
            print ("[DEBUG] EXECUTE CODE - ", exec_str)
            
            if add_image:
                pil_img_path = os.path.join(self.curr_tmp_root_path, self.output_image_path_list[-1])
                if os.path.exists(pil_img_path):
                    pil_img = Image.open(pil_img_path)
                    obs = {
                        "prompt": "\n<|im_start|>user\n" + "<image>" + self.user_prompt.format(exec_str) + "<|im_end|>\n<|im_start|>assistant\n",
                        "multi_modal_data": {"image": [pil_img]},
                    }
                else:
                    obs = "\n<|im_start|>user\n" + self.user_prompt.format(exec_str) + "<|im_end|>\n<|im_start|>assistant\n"
                reward = 0.0
                done = False
                info = {"status": "success", "tool_used": 'code'}
                # return "", reward, done, info
                return obs, reward, done, info
            else:
                obs = "\n<|im_start|>user\n" + self.user_prompt.format(exec_str) + "<|im_end|>\n<|im_start|>assistant\n"
                reward = 0.0
                done = False
                info = {"status": "success", "tool_used": 'code'}
                # return "", reward, done, info
                return obs, reward, done, info

        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)} {action_string=}')
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

        salted_str = str(int(time.time())) + str(random.randint(10000, 99999))
        salted_hash_str = str(hex(hash(salted_str.encode('utf-8')))).split('0x')[-1]
        self.folder_name = salted_hash_str

        self.curr_tmp_root_path = os.path.join(self.root_tmp_path, salted_hash_str)
        os.makedirs(self.curr_tmp_root_path, exist_ok=True)
        pil_img = self.multi_modal_data['image'][0]
        input_image_path = os.path.join(self.curr_tmp_root_path, 'input_image.jpg')
        pil_img.save(input_image_path, format='JPEG')

        self.input_image_path_list = ['input_image.jpg',]
        self.output_image_path_list = []
    
    def delete_env(self):
        if os.path.exists(self.curr_tmp_root_path):
            shutil.rmtree(self.curr_tmp_root_path)
    
    def exec_code(self, code_str):
        if not code_str:
            return None, False
        
        if '```python' not in code_str or '```' not in code_str:
            return 'Code format error.', False
        
        
        add_image = False
        code_str = code_str.replace('```python', '').replace('```', '').strip()
        if 'path_to_input_image.jpg' in code_str:
            code_str = code_str.replace('path_to_input_image.jpg', os.path.join(self.curr_tmp_root_path, self.input_image_path_list[0]))
        if 'path_to_output_image.jpg' in code_str:
            add_image = True
            if len(self.output_image_path_list) == 0:
                output_image_path = '1.jpg'
            else:
                output_image_path = str(len(self.output_image_path_list) + 1) + '.jpg'
            self.output_image_path_list.append(output_image_path)
            code_str = code_str.replace('path_to_output_image.jpg', os.path.join(self.curr_tmp_root_path, self.output_image_path_list[-1]))

        problem = {
            "task_id": "example_task",
            "code": code_str,
            "completion_id": 1,
            "timeout": 30.0
        }

        url_query = random.choice(self.query_urls)

        response = requests.post(url_query, json=problem)

        if response.status_code == 200:
            result = response.json()['output']
            if result.get('passed', 'False'):
                result_str = result.get('result', '')
                return result_str, add_image
            else:
                return 'Code execution error.', False
        else:
            return 'Code execution error.', False


if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualToolBoxV6("visual_toolbox", "Tool for image processing", {})
    
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
<think>To determine the color of the earring, we need to zoom in on the area where the earring is located. This will allow us to see the details more clearly.</think>
<code>
```python
from PIL import Image

# Load the image
image = Image.open('path_to_input_image.jpg')

# Zoom in on the area around the earring
zoomed_in_image = image.crop((40, 60, 230, 480))

# Save the zoomed-in image
zoomed_in_image.save('path_to_output_image.jpg')
```</code>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Result - Reward: {reward}, Info: {info}")
    
