from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from functions import get_mask, canny_image, resize_and_crop, create_pipline
from PIL import Image, ImageOps
import io
import base64
import time
import os
import random
from starlette.responses import JSONResponse


def get_generated_image(pipeline, image_path):
    try:
        mask_image_inner, init_image = get_mask(image_path, 'out')
        mask_image_outer = ImageOps.invert(mask_image_inner)
    except Exception as e:
        print(e)
        mask_image_outer = Image.new('RGB', (512, 512), 'white')

    init_image = resize_and_crop(image_path, 512)
    edges_image = canny_image(init_image)

    prompts = [
        'astronaut floating in the vibrant colors of the cosmos, with high-resolution 8k detail, beautiful outer '
        'space explosion.'
        , "high-detail, 8k image, time traveler, retro clothing, time machine, bygone era."
        , "beautiful, colorful forest scene, Avatar, rendered, 8k detail, glowing flora and big trees."
        , "colorful, high-resolution 8k image, exotic travel destinations, bustling airports, around the world."
        , "vibrant, detailed 8k image, superhero flying above bustling city, cape billowing in the wind."
        , "chilling, 8k image, haunted house at midnight, shadows and ghosts."
        , "beautiful, vibrant 8k image, magical realm with floating islands, glowing plants, and mythical creatures."
        ,
        "grand masquerade ball in an ornate palace, filled with beautifully costumed attendees, rendered in vibrant "
        "colors and high-resolution 8k detail."
        , "stunning, 8k image, glamour and extravagance of the Met Gala, with celebrities, creative outfits."
        , 'vibrant, high-detail 8k image, a surreal landscape filled with optical illusions.'
        , "colorful, highly detailed, cityscape, food and buildings are enlarged to an enormous scale."
        , "high-resolution 8k image, a group of past and present U.S. presidents in a formal gathering."
        , "high-detail 8k image, avantgarde fashion, models in cutting-edge designs, dramatic backdrop."
        , "beautifully detailed 8k image, Game of Thrones, vikings, old english."
        , "magical, vibrant, Harry Potter, Hogwarts castle, and magical creatures."]

    prompt = random.choice(prompts)
    # prompt = "8k, HD, hyper realistic, Harry Potter, magic ,Hogwarts, Voldemort"

    negative_prompt = 'women, female, breast, watermark, text, blur, deformed, bad anatomy, disfigured, poorly drawn ' \
                      'face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, ' \
                      'floating limb, disconnected limb, malformed hands, blurry, ((((mutated hands and finger)))), ' \
                      'watermark, watermarked, over saturated, distorted hands, amputation, missing hands, ' \
                      'double face, obese, doubled face, doubled hands'

    new_image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        image=init_image.convert('RGB'),
        control_image=edges_image.convert('RGB'),
        controlnet_conditioning_scale=0.5,
        mask_image=mask_image_outer.convert('RGB')
    ).images[0]

    return new_image


pipe = create_pipline()

app = FastAPI()

IMAGE_FOLDER = os.path.join(os.getcwd(),'results')
app.mount("/images", StaticFiles(directory=IMAGE_FOLDER), name="images")


@app.post("/image")
async def create_upload_image(image: UploadFile = File(...)):
    global pipe
    init_image = Image.open(image.file)
    gen_image = get_generated_image(pipe, init_image)

    # Convert images to byte array
    byte_arr1 = io.BytesIO()
    time_stamp = time.time() * 100
    gen_url = os.path.join(IMAGE_FOLDER, f"ai{str(time_stamp)}.png")
    gen_image.save(gen_url, format='PNG')
    gen_image.save(byte_arr1, format='PNG')
    byte_arr1 = byte_arr1.getvalue()

    byte_arr1 = base64.b64encode(byte_arr1).decode()

    return JSONResponse(content={
        'image': byte_arr1,
        'URL': f"images/ai{str(time_stamp)}.png"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
