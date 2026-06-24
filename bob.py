# =====================
# IMPORTS
# =====================

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from mne.time_frequency import psd_array_welch

import numpy as np
import pygame
import time
import mne


# =====================
# LOAD EEG DATA
# =====================

raw = mne.io.read_raw_edf(
    "data/S003R03.edf",
    preload=True
)

print(raw.info)


# =====================
# EXTRACT EVENTS
# =====================

events, event_id = mne.events_from_annotations(raw)

print(event_id)


# =====================
# PREPROCESSING
# =====================

raw.filter(8, 30)


# =====================
# CREATE EPOCHS
# =====================

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=0,
    tmax=2,
    baseline=None,
    preload=True
)

epochs = epochs["T1", "T2"]

print(epochs)


# =====================
# CREATE LABELS
# =====================

y = epochs.events[:, -1]

print(y)


# =====================
# SELECT MOTOR CHANNELS
# =====================

motor_channels = [
    "C3..",
    "C4..",
    "Cz..",
    "C1..",
    "C2..",
    "Cp3.",
    "Cp4."
]

epochs_motor = epochs.copy().pick(motor_channels)

print(epochs_motor)


# =====================
# FEATURE EXTRACTION
# =====================

data = epochs_motor.get_data()

psds, freqs = psd_array_welch(
    data,
    sfreq=raw.info["sfreq"],
    fmin=8,
    fmax=30
)

features = np.mean(psds, axis=2)

print("Feature shape:", features.shape)


# =====================
# TRAIN / TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.3,
    random_state=7
)


# =====================
# TRAIN CLASSIFIER
# =====================

model = LinearDiscriminantAnalysis()

model.fit(X_train, y_train)


# =====================
# TEST CLASSIFIER
# =====================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

print("Predictions:", predictions)
print("Actual:", y_test)


# =====================
# CROSS VALIDATION
# =====================

scores = cross_val_score(
    model,
    features,
    y,
    cv=5
)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())


# =====================
# TEXT FEEDBACK
# =====================

for pred in predictions:

    if pred == 2:
        print("MOVE LEFT")

    elif pred == 3:
        print("MOVE RIGHT")

    time.sleep(1)


# =====================
# PYGAME BCI DEMO
# =====================

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((800, 400))

pygame.display.set_caption(
    "EEG Motor Imagery BCI Demo"
)

font = pygame.font.SysFont(
    None,
    48
)

x = 400
y_pos = 200

for pred in predictions:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()

    screen.fill((0, 0, 0))

    if pred == 2:

        x -= 50

        text = font.render(
            "LEFT",
            True,
            (255, 255, 255)
        )

    elif pred == 3:

        x += 50

        text = font.render(
            "RIGHT",
            True,
            (255, 255, 255)
        )

    pygame.draw.rect(
        screen,
        (255, 255, 255),
        (x, y_pos, 50, 50)
    )

    screen.blit(
        text,
        (320, 50)
    )

    pygame.display.update()

    pygame.time.delay(1000)

pygame.time.delay(5000)

pygame.quit()


# =====================
# END PROGRAM
# =====================

input("Press Enter to exit...")
