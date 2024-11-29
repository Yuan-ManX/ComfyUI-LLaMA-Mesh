import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import os
# import spaces
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load the tokenizer and model
model_path = "Zhengyi/LLaMA-Mesh"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

from trimesh.exchange.gltf import export_glb
import gradio as gr
import trimesh
import numpy as np
import tempfile

class ApplyGradientColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_text": ("STRING", {"multiline": True}),  
            }
        }
    
    RETURN_TYPES = ("STRING",)  
    FUNCTION = "apply_gradient_color"
    CATEGORY = "LLaMA-Mesh"

    def apply_gradient_color(self, mesh_text):
        """
        Apply a gradient color to the mesh vertices based on the Y-axis and save as GLB.
        Args:
            mesh_text (str): The input mesh in OBJ format as a string.
        Returns:
            str: Path to the GLB file with gradient colors applied.
        """
        # Load the mesh
        temp_file =  tempfile.NamedTemporaryFile(suffix=f"", delete=False).name
        with open(temp_file+".obj", "w") as f:
            f.write(mesh_text)
        # return temp_file
        mesh = trimesh.load_mesh(temp_file+".obj", file_type='obj')
    
        # Get vertex coordinates
        vertices = mesh.vertices
        y_values = vertices[:, 1]  # Y-axis values
    
        # Normalize Y values to range [0, 1] for color mapping
        y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())
    
        # Generate colors: Map normalized Y values to RGB gradient (e.g., blue to red)
        colors = np.zeros((len(vertices), 4))  # RGBA
        colors[:, 0] = y_normalized  # Red channel
        colors[:, 2] = 1 - y_normalized  # Blue channel
        colors[:, 3] = 1.0  # Alpha channel (fully opaque)
    
        # Attach colors to mesh vertices
        mesh.visual.vertex_colors = colors
    
        # Export to GLB format
        glb_path = temp_file+".glb"
        with open(glb_path, "wb") as f:
            f.write(export_glb(mesh))
        
        return glb_path


class VisualizeMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_text": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",) 
    FUNCTION = "visualize_mesh"
    CATEGORY = "LLaMA-Mesh"

    def visualize_mesh(self, mesh_text):
        """
        Convert the provided 3D mesh text into a visualizable format.
        This function assumes the input is in OBJ format.
        """
        temp_file = "temp_mesh.obj"
        with open(temp_file, "w") as f:
            f.write(mesh_text)
        return temp_file


# @spaces.GPU(duration=120)
def chat_llama3_8b(message: str, 
              history: list, 
              temperature: float, 
              max_new_tokens: int
             ) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.             
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        #print(outputs)
        yield "".join(outputs)
        
