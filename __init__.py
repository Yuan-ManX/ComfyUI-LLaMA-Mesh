from .llama_mesh_nodes import ChatLLaMaMesh, VisualizeMesh, ApplyGradientColor

NODE_CLASS_MAPPINGS = {
    "Apply Gradient Color": ApplyGradientColor,
    "Visualize Mesh": VisualizeMesh,
    "Chat LLaMa Mesh": ChatLLaMaMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyGradientColor": "Apply Gradient Color",
    "VisualizeMesh": "Visualize Mesh",
    "ChatLLaMaMesh": "Chat LLaMa Mesh"
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
