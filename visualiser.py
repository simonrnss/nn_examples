import csv
import pygame
import numpy as np
import pandas as pd
import pygame_gui

data = pd.read_csv('data.csv')

X = data.values

with open('params.csv', 'r') as f:
    reader = csv.reader(f)
    w0, w1, w2 = next(reader)

print(X)
print(w0, w1, w2)

w0 = float(w0)
w1 = float(w1)
w2 = float(w2)

MAX_X = 600
MAX_Y = 600

pygame.init()

pygame.display.set_caption('NN viz')
window_surface = pygame.display.set_mode((MAX_X, MAX_Y))

background = pygame.Surface((MAX_X, MAX_Y))
background.fill(pygame.Color('#000000'))

manager = pygame_gui.UIManager((MAX_X, MAX_Y))

clock = pygame.time.Clock()

is_running = True

x_r = [-4, 6]
y_r = [-2, 6]

def map_pos(x, y):
    nx = MAX_X * (x - x_r[0]) / (x_r[1] - x_r[0])
    ny = MAX_Y - (MAX_Y * (y - y_r[0]) / (y_r[1] - y_r[0]))
    return nx, ny

def unmap(nx, ny):
    y = (-(ny - MAX_Y) * (y_r[1] - y_r[0]) / MAX_Y) + y_r[0]
    x = (nx / MAX_X) * (x_r[1] - x_r[0]) + x_r[0]
    return x, y

for i, row in enumerate(X):
    nx, ny = map_pos(row[0], row[1])
    if i < 50:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    pygame.draw.circle(background, color, (nx, ny), 5)


myfont = pygame.font.Font(None, 20)
old_text = None
old_xbeta = None
old_sig = None
while is_running:
    time_delta = clock.tick(60) / 1000.0
    x_pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
    
        manager.process_events(event)

    manager.update(time_delta)
    
    ox, oy = unmap(x_pos[0], x_pos[1])
    if old_text is not None:
        label = myfont.render(old_text, 0, (0, 0, 0))
        background.blit(label, (50, 50))
    old_text = f"x1 = {ox:.2f}, x2 = {oy:.2f}"
    label = myfont.render(old_text, 0, (255, 255, 255))
    background.blit(label, (50, 50))

    if old_xbeta is not None:
        label = myfont.render(old_xbeta, 0, (0, 0, 0))
        background.blit(label, (50, 75))
    
    
    
    xbeta = w0 + w1 * ox + w2 * oy
    old_xbeta = f"xbeta = {xbeta:.2f}"
    label = myfont.render(old_xbeta, 0, (255, 255, 255))
    background.blit(label, (50, 75))

    if old_sig is not None:
        label = myfont.render(old_sig, 0, (0, 0, 0))
        background.blit(label, (30, 100))

    sig = 1 / (1 + np.exp(-xbeta))
    old_sig = f"1/(1+exp(-xbeta)) = prob of class red = {sig:.2f}"
    label = myfont.render(old_sig, 0, (255*sig, 50, 255*(1-sig)))
    background.blit(label, (30, 100))
    
    


    window_surface.blit(background, (0, 0))
    manager.draw_ui(window_surface)

    pygame.display.update()