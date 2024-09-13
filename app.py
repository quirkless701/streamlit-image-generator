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
st.set_page_config(page_title="YouTube Thumbnail Generator", layout="wide")
st.title("**YouTube Thumbnail Generator**")

# Add a sleek and fun description
st.write("""
    Transform your text prompts into eye-catching YouTube thumbnails with just a few clicks! 
    Whether you need a bold title or a captivating background, our AI-powered generator has you covered. 
    Simply enter your prompt and watch your ideas come to life.
""")

# Input fields for prompt and parameters
prompt = st.text_input("Enter a text prompt", "")
width = st.number_input("Width", min_value=256, max_value=2048, value=1280)
height = st.number_input("Height", min_value=256, max_value=2048, value=720)
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5)
num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=100, value=50)

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate the image
            image = generate_image_from_prompt(prompt, width, height, guidance_scale, num_inference_steps)
            
            # Convert image to bytes for display
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_img = buf.getvalue()
            
            # Display the image
            st.image(byte_img, caption="Generated Image", use_column_width=True, width=700)
            
            # Fullscreen button
            if st.button("View Fullscreen"):
                st.write(f'<img src="data:image/png;base64,{byte_img.encode("base64").decode()}" style="width: 100%; height: 100%;">', unsafe_allow_html=True)
    else:
        st.error("Please enter a prompt to generate an image.")
