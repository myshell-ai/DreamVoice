from dreamvoice import DreamVoice

# Plugin mode (DreamVG + ReDiffVC)
# Initialize DreamVoice in plugin mode with CUDA device
dreamvoice = DreamVoice(mode='plugin', device='cuda')
# Description of the target voice
prompt = 'young female voice, sounds young and cute'
# Provide the path to the content audio and generate the converted audio
gen_audio, sr = dreamvoice.genvc('examples/test1.wav', prompt)
# Save the converted audio
dreamvoice.save_audio('gen1.wav', gen_audio, sr)

# Save the speaker embedding if you like the generated voice
dreamvoice.save_spk_embed('voice_stash1.pt')
# Load the saved speaker embedding
dreamvoice.load_spk_embed('voice_stash1.pt')
# Use the saved speaker embedding for another audio sample
gen_audio2, sr = dreamvoice.simplevc('examples/test2.wav', use_spk_cache=True)
dreamvoice.save_audio('gen2.wav', gen_audio2, sr)


# End-to-end mode (DreamVC)
# Initialize DreamVoice in end-to-end mode with CUDA device
dreamvoice = DreamVoice(mode='end2end', device='cuda')
# Provide the path to the content audio and generate the converted audio
gen_end2end, sr = dreamvoice.genvc('examples/test1.wav', prompt)
# Save the converted audio
dreamvoice.save_audio('gen_end2end.wav', gen_end2end, sr)

# Note: End-to-end mode does not support saving speaker embeddings
# To use a voice generated in end-to-end mode, switch back to plugin mode
# and extract the speaker embedding from the generated audio
# Switch back to plugin mode
dreamvoice = DreamVoice(mode='plugin', device='cuda')
# Load the speaker audio from the previously generated file
gen_end2end2, sr = dreamvoice.simplevc('examples/test2.wav', speaker_audio='gen_end2end.wav')
# Save the new converted audio
dreamvoice.save_audio('gen_end2end2.wav', gen_end2end2, sr)


# Traditional VC
# Plugin mode can be used for traditional one-shot voice conversion
dreamvoice = DreamVoice(mode='plugin', device='cuda')
# Generate audio using traditional one-shot voice conversion
gen_tradition, sr = dreamvoice.simplevc('examples/test1.wav', speaker_audio='examples/speaker.wav')
# Save the converted audio
dreamvoice.save_audio('gen_tradition.wav', gen_tradition, sr)
