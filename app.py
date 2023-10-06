import os
import requests
import random
import gradio as gr
import numpy as np
import PIL.Image
import torch
from typing import List
from diffusers.utils import numpy_to_pil
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from previewer.modules import Previewer
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import htmls

DESCRIPTION = "# 한(韓) - 이미지 생성기 (초간단 버전)"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"
PREVIEW_IMAGES = True



dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    prior_pipeline = WuerstchenPriorPipeline.from_pretrained("warp-ai/wuerstchen-prior", torch_dtype=dtype)
    decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained("warp-ai/wuerstchen", torch_dtype=dtype)
    if ENABLE_CPU_OFFLOAD:
        prior_pipeline.enable_model_cpu_offload()
        decoder_pipeline.enable_model_cpu_offload()
    else:
        prior_pipeline.to(device)
        decoder_pipeline.to(device)

    if USE_TORCH_COMPILE:
        prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
        decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)
    
    if PREVIEW_IMAGES:
        file_path = "text2img_wurstchen_b_v1_previewer_100k.pt"
        url = "https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/text2img_wurstchen_b_v1_previewer_100k.pt"

        if not os.path.exists(file_path):
            response = requests.get(url, allow_redirects=True)
            with open(file_path, 'wb') as file:
                file.write(response.content)

        previewer = Previewer()
        previewer.load_state_dict(torch.load(file_path)["state_dict"])
        previewer.eval().requires_grad_(False).to(device).to(dtype)

        def callback_prior(i, t, latents):
            output = previewer(latents)
            output = numpy_to_pil(output.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            return output
    else:
        previewer = None
        callback_prior = None
else:
    prior_pipeline = None
    decoder_pipeline = None

def check_sensitive_words(text):
    sensitive_words = [    "nude", "nudes", "naked", "bare", "sex", "sexual", "explicit", "adult", "xxx",     "porn", "erotic", "erotica", "sensual", "intimate", "jacuzzi", "hot tub", "taking off",    "undress", "stripping", "breasts", "boobs", "busty", "big breast", "big breasts",     "pussy", "vulva", "genital", "genitals", "private parts", "privates", "explicit content",     "obscene", "lewd", "nudity", "exposed", "topless", "bottomless", "braless", "lingerie",     "bikini", "thong", "g-string", "underwear", "panties", "nipple", "nipples", "areola",     "playboy", "hentai", "xrated", "x-rated", "nude art", "seductive", "inviting", "revealing",    "uncovered", "unclothed", "unveiling", "provocative", "risque"]
    for word in sensitive_words:
        if word in text:
            return True
    return False

# Translate from Korean to English
def translate_to_english(text):
    URI = "https://aic-api.ap.ngrok.io/translate-en/"
    try:
        response = requests.post(URI, json={"text": text})
        if response.status_code == 200:
            return response.json()["translated_text"]
    except Exception as e:
        print("Error during the request:", e)
        return None


# Translate from English to Korean
def translate_to_korean(text):
    URI = "https://aic-api.ap.ngrok.io/translate/"
    try:
        response = requests.post(URI, json={"text": text})
        if response.status_code == 200:
            return response.json()["translated_text"]
    except Exception as e:
        print("Error during the request:", e)
        return None

# Enhance the prompt with additional details
def generate_enhanced_prompt_korean(original_prompt_korean):
    # Translate the original prompt to English
    original_prompt_english = translate_to_english(original_prompt_korean)

    if not original_prompt_english:
        return "Error translating to English", ""

    # Enhance the prompt
    URI = "https://aic-textgen.ap.ngrok.io/api/v1/generate"
    prompt = f"""
Provide a brief description or prompt for a specific image you have in mind, and I will enhance the quality of your prompt by adding descriptive words and artistic nuances to capture the mood and atmosphere of the image. Your enhanced prompt will be in English and will consist of distinct words, adjectives, or short phrases separated by commas, summarized in about three lines.

example)
Original Prompt: plant queen
Revised Prompt: cinematic still, filmed by Guillermo del Toro, Amidst a deep dark forest, an enigmatic being appears--an amalgamation of flora and fauna, with vines for hair, eyes gleaming like embers, and skin adorned with iridescent scales.

Prompt : {original_prompt_english}
Revised Prompt :
"""

    requestData = {
        "prompt": prompt,
        "max_new_tokens": 2000,
        "preset": "None",
        "do_sample": True,
        "temperature": 1.31,
        "top_p": 0.14,
        "typical_p": 1,
        "epsilon_cutoff": 0,
        "eta_cutoff": 0,
        "tfs": 1,
        "top_a": 0,
        "repetition_penalty": 1.17,
        "repetition_penalty_range": 0,
        "top_k": 49,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 4096,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    try:
        response = requests.post(URI, json=requestData)
        if response.status_code == 200:
            enhanced_prompt_english = response.json()["results"][0]["text"].strip()
            # Translate the enhanced prompt back to Korean
            enhanced_prompt_korean = translate_to_korean(enhanced_prompt_english)
            return enhanced_prompt_korean
    except Exception as e:
        print("Error during the request:", e)
        return None, None

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    prior_num_inference_steps: int = 60,
    # prior_timesteps: List[float] = None,
    prior_guidance_scale: float = 4.0,
    decoder_num_inference_steps: int = 12,
    # decoder_timesteps: List[float] = None,
    decoder_guidance_scale: float = 0.0,
    num_images_per_prompt: int = 1,
) -> PIL.Image.Image:
    translate_prompt = translate_to_english(prompt)
    translate_nagative_prompt = translate_to_english(negative_prompt) + "text, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    if check_sensitive_words(translate_prompt):  # Check the translated prompt for sensitive words
        return gr.HTML(value='<div style="color:red;">민감한 단어가 포함되어 있습니다. 다른 단어를 사용해주세요.</div>')
   
    generator = torch.Generator().manual_seed(seed)

    prior_output = prior_pipeline(
        prompt=translate_prompt,
        height=height,
        width=width,
		num_inference_steps  = prior_num_inference_steps,
        # timesteps=DEFAULT_STAGE_C_TIMESTEPS,
        negative_prompt=negative_prompt,
        guidance_scale=prior_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        callback=callback_prior,
    )

   
    decoder_output = decoder_pipeline(
        image_embeddings=prior_output.image_embeddings,
        prompt=translate_prompt,
		num_inference_steps = decoder_num_inference_steps,
        # timesteps=decoder_timesteps,
        guidance_scale=decoder_guidance_scale,
        negative_prompt=translate_nagative_prompt,
        generator=generator,
        output_type="pil",
    ).images
    yield decoder_output

with open("style.css", "r", encoding="utf-8") as css_file:
    css_content = css_file.read()

with gr.Blocks(css=css_content) as demo:
    gr.Markdown(DESCRIPTION)
    
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=2,
                placeholder="이미지를 설명하세요(한글)",
                container=False,
                elem_classes=["custom-textarea"],  # CSS 클래스 추가

            )
        with gr.Row():

            magic_button = gr.Button("매직", scale=1)
            def magic_prompt_clicked(prompt):                
                return generate_enhanced_prompt_korean(prompt)
            magic_button.click(fn=magic_prompt_clicked, inputs=[prompt], outputs=[prompt])
            run_button = gr.Button("생성", scale=1)
        result = gr.Gallery(label="Result", show_label=False, elem_classes=["custom-gallery"])
        with gr.Row():
            negative_prompt = gr.Text(
            label="Negative Prompt",
            show_label=False,
            max_lines=1,
            placeholder="부정 프롬프트를 입력하세요(한글)",
        )
    with gr.Group():
        gr.HTML(value=htmls.title_html_content)
    with gr.Accordion("Advanced options", open=False, visible=False):
        

        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=1024,
                maximum=MAX_IMAGE_SIZE,
                step=512,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=1024,
                maximum=MAX_IMAGE_SIZE,
                step=512,
                value=1024,
            )
            num_images_per_prompt = gr.Slider(
                label="Number of Images",
                minimum=1,
                maximum=20,
                step=1,
                value=1,
            )
        with gr.Row():
            prior_guidance_scale = gr.Slider(
                label="Prior Guidance Scale",
                minimum=0,
                maximum=40,
                step=0.1,
                value=4.0,
            )
            prior_num_inference_steps = gr.Slider(
                label="Prior Inference Steps",
                minimum=1,
                maximum=240,
                step=1,
                value=30,
            )

            decoder_guidance_scale = gr.Slider(
                label="Decoder Guidance Scale",
                minimum=0,
                maximum=20,
                step=0.1,
                value=0.0,
            )
            decoder_num_inference_steps = gr.Slider(
                label="Decoder Inference Steps",
                minimum=1,
                maximum=240,
                step=1,
                value=12,
            )
    

    inputs = [
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            prior_num_inference_steps,
            # prior_timesteps,
            prior_guidance_scale,
            decoder_num_inference_steps,
            # decoder_timesteps,
            decoder_guidance_scale,
            num_images_per_prompt,
    ]
    prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name="run",
    )
    negative_prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    
    run_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )

if __name__ == "__main__":
    demo.launch(favicon_path='aic_logo_ico.png',share=True,enable_queue=True)
