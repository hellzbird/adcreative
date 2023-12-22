
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw, ImageFont,ImageOps
import streamlit as st


def generate_image(url,color):
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    image = download_image(url)
    prompt = f"create a similar image, change the object color to {color}"
    images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1).images
    generated_image = images[0].save("generated_image.jpg")
    return images[0]


def download_image(url):
    image = Image.open(url)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def create_frame_with_resized_image(frame_size, image,logo_path, output_path, new_image_size,logo_size,text,button,color):
    # Create an empty white frame
    frame = Image.new("RGB", frame_size, "white")

    # Open the image to be placed in the frame
    original_image = image
    logo = Image.open(logo_path)
    # Resize the image to the desired dimensions
    resized_image = original_image.resize(new_image_size)
    resized_logo = logo.resize(logo_size)
    # Calculate the position to center the resized image in the frame
    x_position = (frame.width - resized_image.width) // 2
    y_position = (frame.height - resized_image.height) // 2
    
    frame.paste(resized_logo,(200,0))
    # Paste the resized image onto the frame
    frame.paste(resized_image, (x_position, y_position))

    draw = ImageDraw.Draw(frame)
    font_size = 24
    font = ImageFont.truetype("arial.ttf", font_size)
    text_width, text_height = draw.textsize(text, font)
    text_x = (frame.width - text_width) // 2
    
    button_width, button_height = draw.textsize(button, font)
    button_x = (frame.width - button_width) // 2
    button_y = 500
    # Add text to the image
    draw.text((text_x, 450), text, font=font, fill= f"{color}")
    
    # Save the result
    
    frame_thickness = 5
    draw.rectangle([(button_x - frame_thickness, button_y - frame_thickness),
                    (button_x + button_width + frame_thickness, button_y + button_height + frame_thickness)],
                    fill=f"{color}")
    
    draw.text((button_x,500),button,font=font, fill= "white")

    frame.save(output_path)
    

def main():
    st.set_page_config(page_title="img 2 audio story")
    st.header("Turn img into ad template")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    color_input = st.text_input("Enter the color you want in the image")
    st.write("You entered:",color_input)
    
    if uploaded_file is not None and color_input is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption="Uploaded Image.",
                use_column_width=True)
        
        generated_image = generate_image(uploaded_file.name,color_input)
        st.image(generated_image,caption="Generated Image.", use_column_width=True)
    
    
    uploaded_logo = st.file_uploader("Upload the logo", type="jpg")
    #prompt_input = st.text_input("Enter the prompt:")
    #st.write("You entered:", prompt_input)
    punchline_input = st.text_input("Enter the punchline:")
    st.write("You entered:", punchline_input)
    
    button_input = st.text_input("Enter the button text:")
    st.write("You entered:", button_input)
    
    button_color_input = st.text_input("Enter the text and button color:")
    st.write("You entered:",button_color_input)
    
    if uploaded_logo is not None and button_color_input is not None and button_input is not None and punchline_input is not None:
        frame_size = (600, 600)
        logo_data = uploaded_logo.getvalue()
        with open(uploaded_logo.name,  "wb") as file:
            file.write(logo_data)
            
        output_path = "result.jpg"
        new_image_size = (400, 300)  # Adjust these dimensions as needed
        logo_size = (200,150)
        image = generated_image
        logo_path = uploaded_logo.name
        create_frame_with_resized_image(frame_size, image,logo_path, output_path, new_image_size,logo_size,punchline_input,button_input,button_color_input)
        ad_template = Image.open(output_path)
        st.image(ad_template,caption="Generated Ad Template", use_column_width=True)
        
if __name__ == "__main__":
    main()


        
