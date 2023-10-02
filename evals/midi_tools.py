import numpy as np
import pandas as pd
import fortepyan as ff
from matplotlib import pyplot as plt


def to_fortepyan_midi(
    pitch: np.ndarray,
    dstart: np.ndarray,
    duration: np.ndarray,
    velocity: np.ndarray,
) -> ff.MidiPiece:
    # change dstart to start, create end
    start = []
    start.append(dstart[0])
    for i in range(1, len(dstart)):
        start.append(start[i - 1] + dstart[i])

    end = []
    for i in range(len(start)):
        end.append(start[i] + duration[i])

    # pandas dataframe with pitch, start, end, velocity
    df = pd.DataFrame({"pitch": pitch, "start": start, "duration": duration, "end": end, "velocity": velocity})

    piece = ff.MidiPiece(df=df)

    return piece


def plot_piano_roll(piece: ff.MidiPiece, title: str = "Piano Roll") -> plt.Figure:
    fig = ff.view.draw_pianoroll_with_velocities(piece, title=title)
    return fig
