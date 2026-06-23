from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from mne.time_frequency import psd_array_welch
from sklearn.model_selection import cross_val_score


import pygame
import time
import mne

raw = mne.io.read_raw_edf("data/S003R03.edf", preload=True)

print(raw.info)

# raw.plot()

events, event_id = mne.events_from_annotations(raw)

print(event_id)

# mne.viz.plot_events(events)

raw.filter(8, 30)

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

X = epochs.get_data()

print(X.shape)

y = epochs.events[:, -1]

print(y)

# epochs.plot()

motor_channels = [
    "C3..", "C4..", "Cz..",
    "C1..", "C2..",
    "Cp3.", "Cp4."
]
epochs_motor = epochs.copy().pick(motor_channels)

print(epochs_motor)

import numpy as np

data = epochs_motor.get_data()

psds, freqs = psd_array_welch(
    data,
    sfreq=raw.info['sfreq'],
    fmin=8,
    fmax=30
)

features = np.mean(psds, axis=2)

print(features.shape)
print(features[:5])

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.3,
    random_state=7
)

model = LinearDiscriminantAnalysis()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("Predictions:", predictions)
print("Actual:", y_test)

scores = cross_val_score(model, features, y, cv=5)

print(scores)
print("Mean accuracy:", scores.mean())

for pred in predictions:
    if pred == 2:
        print("MOVE LEFT")
    elif pred == 3:
        print("MOVE RIGHT")

    time.sleep(1)

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((800, 400))

pygame.display.set_caption("EEG BCI Demo")

font = pygame.font.SysFont(None, 48)

x = 400
y = 200

for pred in predictions:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    screen.fill((0, 0, 0))

    if pred == 2:
        x -= 50
        text = font.render("LEFT", True, (255, 255, 255))

    elif pred == 3:
        x += 50
        text = font.render("RIGHT", True, (255, 255, 255))

    pygame.draw.rect(screen, (255, 255, 255), (x, y, 50, 50))

    screen.blit(text, (320, 50))

    pygame.display.update()

    pygame.time.delay(1000)

pygame.time.delay(5000)

pygame.quit()

input("Press Enter to exit...")