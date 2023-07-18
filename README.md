# image-gen-diffusers

Diffusers and dlib API
## Description

This project is an API created using FastAPI and uvicorn, two well-known libraries for backend development with Python. The core of the Deep Learning model used in this project is Diffusers, specifically the OltiTest-inpainting-0.0.1 model. This model is a combination of the base Diffusers Inpainting model "sd-v1-5-inpainting.ckpt" and the model "dreamshaper_5BakedVae.safetensors" achieved through Stable Diffusion.

Additionally, the "dlib" library is used, which is utilized for Computer Vision models. In this project, we use the "shape_predictor_68_face_landmarks" model, which detects facial landmarks in a given photo for multiple faces.

Once the faces are identified in the input photo, we generate a modified photo based on the given "Prompts" using the ControlNet Canny algorithm (which creates lines in the photo) and the "controlnet and inpaint diffusers pipeline." In this process, the denoising of the face portion is minimized, ensuring minimal alteration to the facial features.
## Installation

1. Clone the repository: `git clone https://github.com/your/repository.git`
2. Navigate to the project directory:` cd project-directory`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Run the API server: `python main.py`
2. Open your web browser and visit http://localhost:8000 to access the API documentation.
3. Use the provided endpoints to interact with the Diffusers and dlib functionalities.

## Endpoints

The following endpoints are available:

- POST `/image`: Upload an image file and receive a modified image generated based on prompts.

## API Functionality
`/image` Endpoint

This endpoint allows you to upload an image and obtain a modified version of the image generated based on prompts. The image processing pipeline involves the following steps:

The uploaded image is processed to obtain a mask and an initial image.
Facial landmarks are detected using the dlib library.
Canny edge detection is performed on the initial image using the ControlNet Canny algorithm.
A prompt is randomly chosen from a predefined list.
The pipeline generates a modified image based on the prompt, negative prompt, number of inference steps, initial image, control image (edges), and the mask image.
The modified image is saved and returned as a response.

## Contributing

Contributions to this project are welcome. If you want to contribute, please follow these steps:

    Fork the repository.
    Create a new branch: git checkout -b feature-branch
    Make your changes and commit them: git commit -m 'Add some feature'
    Push to the branch: git push origin feature-branch
    Open a pull request.

## License

This project is licensed under the MIT License.
Contact

If you have any questions or feedback regarding this project, please feel free to contact us at [olti@roka.dev].
