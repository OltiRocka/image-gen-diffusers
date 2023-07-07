# Imports
import os.path
import cv2
import dlib
from PIL import Image
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)
from pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline

# Resize to minimum width or height to 512px for better results
def resize_and_crop(image, size=512):
    return image.resize((int(round(image.width / min(image.width, image.height) * size)),
                         int(round(image.height / min(image.width, image.height) * size))))


# Initialize dlib's face detector and the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              'model-zoo\\shape_predictor_68_face_landmarks.dat'))

# Define the indices of the outer facial contour
outer_face_contour_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19,
                              18, 17]


def get_mask(image, mask_type):
    def detect_face(img):
        # Detect faces in the image
        return detector(img, 1)

    def get_face_landmarks(img, detected):
        # Get the landmarks/parts for the face in box d.
        return [predictor(img, detection) for detection in detected]

    def create_face_mask(img, face_landmarks, type_mask):
        # Initialize an empty mask with the same size as the input image
        init_mask = np.zeros(img.shape, dtype=np.uint8)

        for face_landmarks in face_landmarks:
            # Get the outer facial contour points
            outer_facial_contour = [face_landmarks.part(i) for i in outer_face_contour_indices]

            # Convert the outer facial contour points to image coordinates
            outer_facial_contour = [(pt.x, pt.y) for pt in outer_facial_contour]

            # Draw a white filled polygon on the mask using the outer facial contour points
            if type_mask == 'out':
                cv2.fillPoly(init_mask, [np.array(outer_facial_contour, dtype=np.int32)], (255, 255, 255))
            elif type_mask == 'in':
                init_mask.fill(255)
                cv2.fillPoly(init_mask, [np.array(outer_facial_contour, dtype=np.int32)], (0, 0, 0))

        return init_mask

    # Load an image

    image = image.convert("RGB")
    image = np.array(image)
    # Detect faces in the image
    detections = detect_face(image)
    # Get face landmarks for the detected faces
    landmarks = get_face_landmarks(image, detections)
    # Create a face mask using the landmarks
    mask = create_face_mask(image, landmarks, mask_type)

    mask_image = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    init_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    mask_image = resize_and_crop(mask_image, 512)
    init_image = resize_and_crop(init_image, 512)
    return mask_image, init_image

# Create the canny image for ControlNet Model
def canny_image(image_path):
    opencv_image = np.array(image_path)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv2.Canny(img, 20, 50)

    edges_image = Image.fromarray(edges)
    edges_image = resize_and_crop(edges_image, 512)
    return edges_image

# Create main pipeline for image generation
def create_pipline():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        os.path.join(os.getcwd(),'model-zoo','OltiTest-inpainting-0.0.1'), revision="fp16", controlnet=controlnet,
        torch_dtype=torch.float16
    ).to('cuda')

    # Turn off NSFW filter for generated images
    def dummy_checker(images, **kwargs): return images, False

    pipe.safety_checker = dummy_checker
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

    return pipe
