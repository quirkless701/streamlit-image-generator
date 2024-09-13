import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model from Hugging Face with appropriate dtype
if device == "cuda":
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
else:
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

pipeline = pipeline.to(device)

def generate_image_from_prompt(prompt, width=1280, height=720, guidance_scale=7.5, num_inference_steps=50):
    """
    Generate an image from a text prompt with configurable parameters.
    
    Args:
    - prompt (str): The text prompt to generate the image from.
    - width (int): The width of the generated image.
    - height (int): The height of the generated image.
    - guidance_scale (float): Controls how much the model follows the text prompt.
    - num_inference_steps (int): The number of steps for diffusion, more steps usually mean better quality.
    
    Returns:
    - Image: The generated PIL image object.
    """
    # Generate image
    with torch.no_grad():
        image = pipeline(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        
        # Resize the image
        image = image.resize((width, height), Image.LANCZOS)
    
    return image

# Streamlit UI
st.title("Text to Image Generator")
prompt = st.text_input("Enter a text prompt", "")

# Additional options for the user to customize
width = st.slider("Select image width", 512, 1920, 1280, step=64)
height = st.slider("Select image height", 512, 1080, 720, step=64)
guidance_scale = st.slider("Select guidance scale", 1.0, 20.0, 7.5)
num_inference_steps = st.slider("Select number of inference steps", 10, 100, 50)

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate the image
            image = generate_image_from_prompt(prompt, width, height, guidance_scale, num_inference_steps)
            
            # Display the image
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter a prompt to generate an image.")
