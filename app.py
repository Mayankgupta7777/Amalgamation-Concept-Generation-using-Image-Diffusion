#!/usr/bin/env python


from __future__ import annotations
import sys
import os
import pathlib

import gradio as gr
import torch

from inference import InferencePipeline
from trainer import Trainer
from uploader import upload

def show_warning(warning_text: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown(warning_text)
    return demo


def update_output_files() -> dict:
    paths = sorted(pathlib.Path('results').glob('*.bin'))
    paths = [path.as_posix() for path in paths]  # type: ignore
    return gr.update(value=paths or None)


def create_training_demo(trainer: Trainer,
                         pipe: InferencePipeline) -> gr.Blocks:
    with gr.Blocks() as demo:
        base_model = gr.Dropdown(
            choices=['stabilityai/stable-diffusion-2-1-base', 'CompVis/stable-diffusion-v1-4'],
            value='CompVis/stable-diffusion-v1-4',
            label='Base Model',
            visible=True)
        resolution = gr.Dropdown(choices=['512', '768'],
                                 value='512',
                                 label='Resolution',
                                 visible=True)

        with gr.Row():
            with gr.Box():
                concept_images_collection = []
                concept_prompt_collection = []
                class_prompt_collection = []
                buttons_collection = []
                delete_collection = []
                is_visible = []
                maximum_concepts = 3
                row = [None] * maximum_concepts
                for x in range(maximum_concepts):
                    ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
                    ordinal_concept = ["<new1> cat", "<new2> wooden pot", "<new3> chair"]
                    if(x == 0):
                        visible = True
                        is_visible.append(gr.State(value=True))
                    else:
                        visible = False
                        is_visible.append(gr.State(value=False))

                    concept_images_collection.append(gr.Files(label=f'''Upload the images for your {ordinal(x+1) if (x>0) else ""} concept''', visible=visible))
                    with gr.Column(visible=visible) as row[x]:
                        concept_prompt_collection.append(
                            gr.Textbox(label=f'''{ordinal(x+1) if (x>0) else ""} concept prompt ''', max_lines=1, 
                                        placeholder=f'''Example: "photo of a {ordinal_concept[x]}"''' )
                            )  
                        class_prompt_collection.append(
                            gr.Textbox(label=f'''{ordinal(x+1) if (x>0) else ""} class prompt ''', 
                                        max_lines=1, placeholder=f'''Example: "{ordinal_concept[x][7:]}"''')
                            )
                    with gr.Row():
                        if(x < maximum_concepts-1):
                            buttons_collection.append(gr.Button(value=f"Add {ordinal(x+2)} concept", visible=visible))
                        if(x > 0):
                            delete_collection.append(gr.Button(value=f"Delete {ordinal(x+1)} concept"))
            
                counter_add = 1
                for button in buttons_collection:
                    if(counter_add < len(buttons_collection)):
                        button.click(lambda:
                        [gr.update(visible=True),gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), True, None],
                        None, 
                        [row[counter_add], concept_images_collection[counter_add], buttons_collection[counter_add-1], buttons_collection[counter_add], is_visible[counter_add], concept_images_collection[counter_add]], queue=False)
                    else:
                        button.click(lambda:
                        [gr.update(visible=True),gr.update(visible=True), gr.update(visible=False), True], 
                        None, 
                        [row[counter_add], concept_images_collection[counter_add], buttons_collection[counter_add-1], is_visible[counter_add]], queue=False)
                    counter_add += 1
                
                counter_delete = 1
                for delete_button in delete_collection:
                    if(counter_delete < len(delete_collection)+1):
                        if counter_delete == 1:
                            delete_button.click(lambda:
                            [gr.update(visible=False, value=None),gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),False], 
                            None, 
                            [concept_images_collection[counter_delete], row[counter_delete], buttons_collection[counter_delete-1], buttons_collection[counter_delete], is_visible[counter_delete]], queue=False)
                        else:
                            delete_button.click(lambda:
                            [gr.update(visible=False, value=None),gr.update(visible=False), gr.update(visible=True), False], 
                            None, 
                            [concept_images_collection[counter_delete], row[counter_delete], buttons_collection[counter_delete-1], is_visible[counter_delete]], queue=False)
                    counter_delete += 1
                gr.Markdown('''
                        - We use "\<new1\>" modifier_token in front of the concept, e.g., "\<new1\> cat". For multiple concepts use "\<new2\>",  "\<new3\>" etc. Increase the number of steps with more concepts.
                        - For a new concept an e.g. concept prompt is "photo of a \<new1\> cat" and "cat" for class prompt.
                        - For a style concept, use "painting in the style of \<new1\> art" for concept prompt and "art" for class prompt.
                        - Class prompt should be the object category.
                        - If "Train Text Encoder", disable "modifier token" and use any unique text to describe the concept e.g. "ktn cat". 
                        ''')
            with gr.Box():
                gr.Markdown('Training Parameters')
                with gr.Row():
                    modifier_token = gr.Checkbox(label='modifier token',
                                                value=True)
                    train_text_encoder = gr.Checkbox(label='Train Text Encoder',
                                            value=False)
                num_training_steps = gr.Number(
                    label='Number of Training Steps', value=1000, precision=0)
                learning_rate = gr.Number(label='Learning Rate', value=0.00001)
                batch_size = gr.Number(
                    label='batch_size', value=1, precision=0)
                with gr.Row():
                    use_8bit_adam = gr.Checkbox(label='Use 8bit Adam', value=True) 
                    gradient_checkpointing = gr.Checkbox(label='Enable gradient checkpointing', value=False)
                with gr.Accordion('Other Parameters', open=False):
                    gradient_accumulation = gr.Number(
                        label='Number of Gradient Accumulation',
                        value=1,
                        precision=0)
                    num_reg_images = gr.Number(
                        label='Number of Class Concept images',
                        value=200,
                        precision=0)
                    gen_images = gr.Checkbox(label='Generated images as regularization',
                                                 value=False)
                gr.Markdown('''
                    - It will take about ~10 minutes to train for 1000 steps and ~21GB on a 3090 GPU. 
                    - Our results in the paper are trained with batch-size 4 (8 including class regularization samples).
                    - Enable gradient checkpointing for lower memory requirements (~14GB) at the expense of slower backward pass.
                    - Note that your trained models will be deleted when the second training is started. You can upload your trained model in the "Upload" tab.
                    - We retrieve real images for class concept using clip_retireval library which can take some time. 
                    ''')

        run_button = gr.Button('Start Training')
        with gr.Box():
            with gr.Row():
                check_status_button = gr.Button('Check Training Status')
                with gr.Column():
                    with gr.Box():
                        gr.Markdown('Message')
                        training_status = gr.Markdown()
                    output_files = gr.Files(label='Trained Weight Files')

        run_button.click(fn=pipe.clear,
                            inputs=None,
                            outputs=None,)
        run_button.click(fn=trainer.run,
                         inputs=[
                             base_model,
                             resolution,
                             num_training_steps,
                             learning_rate,
                             train_text_encoder,
                             modifier_token,
                             gradient_accumulation,
                             batch_size,
                             use_8bit_adam,
                             gradient_checkpointing,
                             gen_images,
                             num_reg_images,
                         ] +
                             concept_images_collection + 
                             concept_prompt_collection +
                             class_prompt_collection 
                         ,
                         outputs=[
                             training_status,
                             output_files,
                         ],
                         queue=False)
        check_status_button.click(fn=trainer.check_if_running,
                                  inputs=None,
                                  outputs=training_status,
                                  queue=False)
        check_status_button.click(fn=update_output_files,
                                  inputs=None,
                                  outputs=output_files,
                                  queue=False)
    return demo


def find_weight_files() -> list[str]:
    curr_dir = pathlib.Path(__file__).parent
    paths = sorted(curr_dir.rglob('*.bin'))
    paths = [path for path in paths if '.lfs' not in str(path)]
    return [path.relative_to(curr_dir).as_posix() for path in paths]


def reload_custom_diffusion_weight_list() -> dict:
    return gr.update(choices=find_weight_files())


def create_inference_demo(pipe: InferencePipeline) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                base_model = gr.Dropdown(
                    choices=['stabilityai/stable-diffusion-2-1-base', 'CompVis/stable-diffusion-v1-4'],
                    value='CompVis/stable-diffusion-v1-4',
                    label='Base Model',
                    visible=True)
                resolution = gr.Dropdown(choices=[512, 768],
                                 value=512,
                                 label='Resolution',
                                 visible=True)
                reload_button = gr.Button('Reload Weight List')
                weight_name = gr.Dropdown(choices=find_weight_files(),
                                               value='custom-diffusion-models/cat.bin',
                                               label='Custom Diffusion Weight File')
                prompt = gr.Textbox(
                    label='Prompt',
                    max_lines=1,
                    placeholder='Example: "\<new1\> cat in outer space"')
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=42)
                with gr.Accordion('Other Parameters', open=False):
                    num_steps = gr.Slider(label='Number of Steps',
                                          minimum=0,
                                          maximum=500,
                                          step=1,
                                          value=100)
                    guidance_scale = gr.Slider(label='CFG Scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=6)
                    eta = gr.Slider(label='DDIM eta',
                                               minimum=0,
                                               maximum=1.,
                                               step=0.1,
                                               value=1.)
                    batch_size = gr.Slider(label='Batch Size',
                                               minimum=0,
                                               maximum=10.,
                                               step=1,
                                               value=1)

                run_button = gr.Button('Generate')

                gr.Markdown('''
                - Models with names starting with "custom-diffusion-models/" are the pretrained models provided in the [original repo](https://github.com/adobe-research/custom-diffusion), and the ones with names starting with "results/delta.bin" are your trained models.
                - After training, you can press "Reload Weight List" button to load your trained model names.
                - Increase number of steps in Other parameters for better samples qualitatively. 
                ''')
            with gr.Column():
                result = gr.Image(label='Result')

        reload_button.click(fn=reload_custom_diffusion_weight_list,
                            inputs=None,
                            outputs=weight_name)
        prompt.submit(fn=pipe.run,
                      inputs=[
                          base_model,
                          weight_name,
                          prompt,
                          seed,
                          num_steps,
                          guidance_scale,
                          eta,
                          batch_size,
                          resolution
                      ],
                      outputs=result,
                      queue=False)
        run_button.click(fn=pipe.run,
                         inputs=[
                             base_model,
                             weight_name,
                             prompt,
                             seed,
                             num_steps,
                             guidance_scale,
                             eta,
                             batch_size,
                             resolution
                         ],
                         outputs=result,
                         queue=False)
    return demo


def create_upload_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        model_name = gr.Textbox(label='Model Name')
        hf_token = gr.Textbox(
            label='Hugging Face Token (with write permission)')
        upload_button = gr.Button('Upload')
        with gr.Box():
            gr.Markdown('Message')
            result = gr.Markdown()
        gr.Markdown('''
            - You can upload your trained model to your private Model repo (i.e. https://huggingface.co/{your_username}/{model_name}).
            - You can find your Hugging Face token [here](https://huggingface.co/settings/tokens).
            ''')

    upload_button.click(fn=upload,
                        inputs=[model_name, hf_token],
                        outputs=result)

    return demo


pipe = InferencePipeline()
trainer = Trainer()

with gr.Blocks(css='style.css') as demo:
    if os.getenv('IS_SHARED_UI'):
        show_warning(SHARED_UI_WARNING)
    if not torch.cuda.is_available():
        show_warning(CUDA_NOT_AVAILABLE_WARNING)

    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    gr.Markdown(DETAILDESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('Train'):
            create_training_demo(trainer, pipe)
        with gr.TabItem('Test'):
            create_inference_demo(pipe)
        with gr.TabItem('Upload'):
            create_upload_demo()

demo.queue(default_enabled=False).launch(share=False)
