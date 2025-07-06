import fal
import logging
from tasks import Text2ImgInput, Img2ImgInput, ImgOutput
from model import FluxPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# is local_python_modules working?
# it is yeah


class MyApp(fal.App, keep_alive=300, name="take-home"):
    machine_type = "GPU-H100"
    requirements = [
        "hf-transfer==0.1.9",
        "diffusers[torch]==0.34.0",
        "transformers[sentencepiece]==4.51.1",
        "accelerate==1.6.0",
        "torch==2.5.1",
        "fal",
    ]
    local_python_modules = [
        "tasks",
        "model",
    ]

    def setup(self):
        logger.info("[DEBUG] MyApp.setup() called")
        self.Text2ImgInput = Text2ImgInput
        self.Img2ImgInput = Img2ImgInput
        self.ImgOutput = ImgOutput
        logger.info("[DEBUG] About to create FluxPredictor")
        self.flux_predictor = FluxPredictor()
        logger.info("[DEBUG] FluxPredictor created, calling warmup")
        self.warmup()

    def warmup(self):
        logger.info("[DEBUG] Warmup method called")
        warmup_task = self.Text2ImgInput(
            prompt="a picture of a cat", negative_prompt=""
        )
        logger.info(f"[DEBUG] Warmup task created: nag_scale={warmup_task.nag_scale}")
        self.flux_predictor.do_text_2_img(warmup_task)

    @fal.endpoint("/")
    def text2img(self, text_2_img_request: Text2ImgInput):
        logger.info(
            f"[DEBUG] text2img endpoint called with nag_scale={text_2_img_request.nag_scale}"
        )
        image = self.flux_predictor.do_text_2_img(text_2_img_request)
        logger.info("[DEBUG] text2img endpoint completed")
        return image

    @fal.endpoint("/img2img")
    def img2img(self, img_2_img_request: Img2ImgInput):
        image = self.flux_predictor.do_img_2_img(img_2_img_request)
        return image


# TODO add a "validate task" method that makes sure the values are nice
# not needed 
