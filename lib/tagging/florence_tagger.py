import logging
import time

import PIL.Image
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from lib.upscaling.util import scale_to_megapixels

logger = logging.getLogger(__name__)


class FlorenceTagger:
    def __init__(self):
        # Load the Florence2 model and processor
        logger.info("Initializing Florence-2 tagger")
        start_time = time.time()

        model_id = 'MiaoshouAI/Florence-2-base-PromptGen-v2.0'
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()

        elapsed = time.time() - start_time
        logger.info(f"Florence-2 model loaded in {elapsed:.2f}s")

    def predict(self, image: Image, prompt: str):
        """
            Generate a caption for an image using the Florence2 model.
            :param image: The image to generate a caption for
            :param prompt: <GENERATE_PROMPT> <DETAILED_CAPTION> <MORE_DETAILED_CAPTION>
            :return:
            """
        logger.info(f"Generating Florence-2 caption with prompt: {prompt}")
        start_time = time.time()

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Generate the output
        gen_start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                output_scores=True,  # Enable this to get the scores
                return_dict_in_generate=True  # Enable this to get the output as a dict
            )
        gen_time = time.time() - gen_start
        logger.debug(f"Florence-2 generation completed in {gen_time:.3f}s")

        generated_ids = outputs.sequences
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Convert the generated text into a caption
        caption = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        result = list(caption.values())[0]
        total_time = time.time() - start_time
        logger.info(f"Florence-2 prediction complete in {total_time:.2f}s: '{result[:100]}...'")

        # Return the caption and probabilities for each token
        return result

def generate_florence_caption(image_path: str, prompt: str = '<GENERATE_PROMPT>') -> str:
    logger.info(f"Generating Florence caption for: {image_path}")
    try:
        image = PIL.Image.open(image_path)
        original_size = image.size
        # resize to 0.5 megapixels
        image = scale_to_megapixels(image, 0.5)
        logger.debug(f"Image resized from {original_size} to {image.size}")

        captioning = FlorenceTagger()
        caption = captioning.predict(image, prompt)
        return caption
    except Exception as e:
        logger.error(f"Error generating Florence caption for {image_path}: {str(e)}", exc_info=True)
        raise
