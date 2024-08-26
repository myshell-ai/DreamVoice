import gradio as gr
from dreamvoice import DreamVoice

# Initialize DreamVoice in end-to-end mode with CUDA device
dreamvoice = DreamVoice(mode='plugin', device='cuda')

# Define the function that will be called by Gradio
def voice_conversion(audio, prompt, prompt_random_seed=None, vc_random_seed=None):
    # Convert seeds to None or int
    prompt_random_seed = int(prompt_random_seed) if prompt_random_seed and prompt_random_seed.isdigit() else None
    vc_random_seed = int(vc_random_seed) if vc_random_seed and vc_random_seed.isdigit() else None

    # Generate the converted audio with the provided random seed
    gen_end2end, sr = dreamvoice.genvc(
        audio, prompt,
        prompt_guidance_scale=3, prompt_guidance_rescale=0.0,
        prompt_ddim_steps=100, prompt_eta=1, prompt_random_seed=prompt_random_seed,
        vc_guidance_scale=3, vc_guidance_rescale=0.0,
        vc_ddim_steps=50, vc_eta=1, vc_random_seed=vc_random_seed
    )
    # Return the converted audio and sampling rate
    return sr, gen_end2end

# Create a Gradio interface
demo = gr.Interface(
    fn=voice_conversion,
    inputs=[
        gr.Audio(label="Input Audio", type="filepath"),  # This returns the file path
        gr.Textbox(label="Prompt (Optional)"),
        gr.Textbox(label="Prompt Random Seed (Optional)", placeholder="Optional: Enter a number to repeat a specific result"),
        gr.Textbox(label="Voice Conversion Random Seed", placeholder="Optional: Enter a number to repeat a specific result")
    ],
    outputs=gr.Audio(label="Converted Audio", type="numpy"),  # This expects a (sr, audio_data) tuple
    title="DreamVoice Demo",
    description="Upload or record an audio file and enter a prompt to generate a voice-converted audio using DreamVoice."
)

# Launch the Gradio demo
demo.launch(share=True)
