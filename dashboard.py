import glob

import torch
import streamlit as st

from model.midi_dcgan import Generator
from utils.checkpoint_utils import load_checkpoint
from evals.midi_tools import denormalize, plot_piano_roll, to_fortepyan_midi, render_midi_to_mp3

st.set_page_config(layout="wide", page_title="MIDI DCGAN", page_icon=":musical_keyboard")


def generate_midi(generator, cfg):
    # create a random noise vector
    noise = torch.randn(1, cfg.generator.noise_size, cfg.data.channels, device=cfg.system.device)
    fake_data = generator(noise).squeeze(0).detach().cpu().numpy()

    dstart = denormalize(fake_data[0, :], 0.0, 3.910354948327446)  # values from preprocess_maestro
    duration = denormalize(fake_data[1, :], 0.001041124508395427, 4.609743047833225)  # values from preprocess_maestro
    velocity = fake_data[2, :] * 127
    pitch = fake_data[3, :] * 127
    # round to nearest integer
    pitch = pitch.round().astype(int)
    velocity = velocity.round().astype(int)

    fortepyan_midi = to_fortepyan_midi(pitch, dstart, duration, velocity)
    return fortepyan_midi


def display_audio(fortepyan_midi, num=0):
    st.title("MIDI generated by DCGAN")
    fig = plot_piano_roll(
        fortepyan_midi,
        title=f"Generation {num}",
    )
    original_mp3_path = render_midi_to_mp3(
        piece=fortepyan_midi,
        filename=f"generation_{num}.mp3",
    )
    st.pyplot(fig)
    st.audio(original_mp3_path, format="audio/mp3", start_time=0)


def main():
    with st.sidebar:
        # Show available checkpoints
        options = glob.glob("checkpoints/*.pt")
        options.sort()
        checkpoint_path = st.selectbox(label="model", options=options)
        st.markdown("Selected checkpoint:")
        st.markdown(checkpoint_path)

    # Load selected checkpoint
    checkpoint, cfg = load_checkpoint(ckpt_path=checkpoint_path, omegaconf=True)
    generator = Generator(
        noise_size=cfg.generator.noise_size,
        output_size=cfg.data.size,
    ).to(cfg.system.device)
    generator.load_state_dict(checkpoint["generator_state_dict"])

    if "num" not in st.session_state:
        st.session_state.num = 0

    if st.button("Generate"):
        midi = generate_midi(generator, cfg)
        display_audio(fortepyan_midi=midi, num=st.session_state.num)
        st.session_state.num += 1


if __name__ == "__main__":
    main()
