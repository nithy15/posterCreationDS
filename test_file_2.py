import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from skimage.measure import label
from io import BytesIO
import os

def process_image(uploaded_file):
    # Parameters
    template_width, template_height = 1600, 1200
    background_color = (200, 230, 255)
    corner_image_path = os.path.join(os.getcwd(), "corner_image.png")  # Path to the corner image
    
    # Create a blank template
    template = Image.new("RGB", (template_width, template_height), background_color)

    # Load the uploaded center image
    center_image = Image.open(uploaded_file).convert("RGBA")

    # Convert center image to an RGBA array
    center_array = np.array(center_image)

    # Define a mask for white areas
    threshold = 230
    white_mask = (center_array[..., :3] > threshold).all(axis=-1)

    # Label connected regions of the white mask
    labeled_mask = label(white_mask)

    # Identify regions touching the edge
    edges = np.zeros_like(labeled_mask, dtype=bool)
    edges[0, :] = edges[-1, :] = edges[:, 0] = edges[:, -1] = True
    background_labels = set(labeled_mask[edges].flatten()) - {0}

    # Create a mask for background white regions
    background_white_mask = np.isin(labeled_mask, list(background_labels))

    # Replace background white areas with the background color
    center_array[..., :3][background_white_mask] = background_color

    # Create a new image from the modified array
    center_image_filled = Image.fromarray(center_array, "RGBA").convert("RGB")

    # Ensure the center image is exactly 1024x1024
    center_image_filled = ImageOps.fit(center_image_filled, (1024, 1024))

    # Calculate the position to paste the center image on the template
    x_offset = (template_width - center_image_filled.width) // 2
    y_offset = (template_height - center_image_filled.height) // 2

    template.paste(center_image_filled, (x_offset, y_offset))

    # Add bottom-right image from the local file in original size
    if os.path.exists(corner_image_path):
        corner_image = Image.open(corner_image_path).convert("RGBA")
        corner_width, corner_height = corner_image.size

        # Resize the corner image to 2.5x its original size
        new_size = (int(corner_width * 2.5), int(corner_height * 2.5))
        corner_image_resized = corner_image.resize(new_size, Image.Resampling.LANCZOS)


        # Calculate bottom-right position
        corner_x = template_width - corner_image_resized.width - 20  # 20px padding
        corner_y = template_height - corner_image_resized.height - 20  # 20px padding

        # Paste the bottom-right image onto the template
        template.paste(corner_image_resized, (corner_x, corner_y), corner_image_resized)


    # Save the result to a BytesIO stream
    img_stream = BytesIO()
    template.save(img_stream, format="PNG")
    img_stream.seek(0)
    return img_stream

# Streamlit UI
st.title("Image Processing App")
st.write("Upload a image to process the poster template.")

# File uploader
uploaded_file = st.file_uploader("Choose the image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded center image
    st.image(uploaded_file, caption="DALLE 3 Image", use_column_width=True)

    # Process the images
    st.write("Processing the image...")
    processed_image_stream = process_image(uploaded_file)

    # Display processed image
    processed_image = Image.open(processed_image_stream)
    st.image(processed_image, caption="Poster Template", use_column_width=True)

    # Provide download option
    st.download_button(
        label="Download Processed Template",
        data=processed_image_stream,
        file_name="poster_template.png",
        mime="image/png"
    )
