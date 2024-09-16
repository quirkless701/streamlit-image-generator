import torch
from diffusers import StableDiffusionPipeline
import streamlit as st
from PIL import Image

# Streamlit app title
st.title("YouTube Thumbnail Generator")

# Model and device setup
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipe.to(device)

pipe = load_model()

# Input prompt from user
prompt = st.text_input("Enter your prompt for the YouTube thumbnail:", 
                       "A beautiful landscape with mountains and sunset")

# Input for custom thumbnail dimensions
st.write("Set the custom thumbnail dimensions (default 1280x720 for YouTube)")
width = st.number_input("Enter width (px):", value=1280)
height = st.number_input("Enter height (px):", value=720)

# Generate the thumbnail when button is clicked
if st.button("Generate Thumbnail"):
    with st.spinner("Generating thumbnail..."):
        # Generate image using Stable Diffusion
        image = pipe(prompt).images[0]
        
        # Resize the image to user-specified dimensions
        thumbnail_size = (int(width), int(height))
        image = image.resize(thumbnail_size)

        # Display the resized image in Streamlit
        st.image(image, caption=f"Generated Thumbnail: {prompt}", use_column_width=True)

        # Save the image as a PNG file
        image.save("youtube_thumbnail.png")
        st.success("Thumbnail saved as youtube_thumbnail.png")

        # Display download button
        with open("youtube_thumbnail.png", "rb") as file:
            btn = st.download_button(
                label="Download Thumbnail",
                data=file,
                file_name="youtube_thumbnail.png",
                mime="image/png"
            )
