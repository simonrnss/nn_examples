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
MAX_Y = 800
MAX_Y_PLOT = 600
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
    ny = MAX_Y_PLOT - (MAX_Y_PLOT * (y - y_r[0]) / (y_r[1] - y_r[0]))
    return nx, ny

def unmap(nx, ny):
    y = (-(ny - MAX_Y_PLOT) * (y_r[1] - y_r[0]) / MAX_Y_PLOT) + y_r[0]
    x = (nx / MAX_X) * (x_r[1] - x_r[0]) + x_r[0]
    return x, y

for i, row in enumerate(X):
    nx, ny = map_pos(row[0], row[1])
    if i < 50:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    pygame.draw.circle(background, color, (nx, ny), 5)

def draw_network(x1, x2, background):
    x1text = myfont.render("x1", 0, (255, 255, 255))
    background.blit(x1text, (10, MAX_Y_PLOT + 15))
    x1text = myfont.render("x2", 0, (255, 255, 255))
    background.blit(x1text, (10, MAX_Y_PLOT + 105))
    pygame.draw.rect(
        background,
        (255, 255, 255),
        pygame.Rect(30, MAX_Y_PLOT + 10, 50, 30)
    )
    pygame.draw.rect(
        background,
        (255, 255, 255),
        pygame.Rect(30, MAX_Y_PLOT + 100, 50, 30)
    )
    x1text = myfont.render(f"{x1:.1f}", 0, (0, 0, 0))
    background.blit(x1text, (35, MAX_Y_PLOT + 15))
    x2text = myfont.render(f"{x2:.1f}", 0, (0, 0, 0))
    background.blit(x2text, (35, MAX_Y_PLOT + 105))


    pygame.draw.rect(
        background,
        (50, 50, 50),
        pygame.Rect(180, MAX_Y_PLOT + 30, 170, 50)
    )
    pygame.draw.rect(
        background,
        (255, 255, 255),
        pygame.Rect(205, MAX_Y_PLOT + 40, 40, 30)
    )

    xbeta = w0 + w1 * ox + w2 * oy
    xbtext = myfont.render(f"{xbeta:.1f}", 0, (0, 0, 0))
    background.blit(xbtext, (210, MAX_Y_PLOT + 50))

    pred = 1 / (1 + np.exp(-xbeta))

    pygame.draw.rect(
        background,
        (255*pred, 50, 255*(1-pred)),
        pygame.Rect(275, MAX_Y_PLOT + 40, 50, 30)
    )

    ptext = myfont.render(f"{pred:.2f}", 0, (255, 255, 255))
    background.blit(ptext, (280, MAX_Y_PLOT + 50))

    text = myfont.render("xbeta", 0, (255, 255, 255))
    background.blit(text, (210, MAX_Y_PLOT+80))

    text = myfont.render("p(red) = 1/(1 + exp(-xbeta))", 0, (255, 255, 255))
    background.blit(text, (280, MAX_Y_PLOT+80))

    pygame.draw.line(
        background,
        (255, 255, 255),
        (55, MAX_Y_PLOT + 115),
        (175, MAX_Y_PLOT + 55)
    )

    pygame.draw.line(
        background,
        (255, 255, 255),
        (55, MAX_Y_PLOT + 15),
        (175, MAX_Y_PLOT + 55)
    )

    pygame.draw.line(
        background,
        (255, 255, 255),
        (245, MAX_Y_PLOT + 55),
        (275, MAX_Y_PLOT + 55)
    )


    pygame.draw.rect(
        background,
        (255, 255, 255),
        pygame.Rect(100, MAX_Y_PLOT + 20, 40, 30)
    )

    x1b = x1*w1
    x1btext = myfont.render(f"{x1b:.2f}", 0, (0, 0, 0))
    background.blit(x1btext, (105, MAX_Y_PLOT+25))

    background.blit(
        myfont.render("x1*beta1", 0, (255, 255, 255)),
        (105, MAX_Y_PLOT + 5)
    )

    pygame.draw.rect(
        background,
        (255, 255, 255),
        pygame.Rect(100, MAX_Y_PLOT + 70, 40, 30)
    )

    x2b = x2*w2
    x1btext = myfont.render(f"{x2b:.2f}", 0, (0, 0, 0))
    background.blit(x1btext, (105, MAX_Y_PLOT+75))

    background.blit(
        myfont.render("x2*beta2", 0, (255, 255, 255)),
        (105, MAX_Y_PLOT + 110)
    )


    pygame.draw.rect(
        background,
        (255, 255, 255),
        pygame.Rect(120, MAX_Y_PLOT + 140, 40, 30)
    )

    x0b = w0
    x0btext = myfont.render(f"{x0b:.2f}", 0, (0, 0, 0))
    background.blit(x0btext, (125, MAX_Y_PLOT+145))

    background.blit(
        myfont.render("beta0", 0, (255, 255, 255)),
        (125, MAX_Y_PLOT + 180)
    )

    pygame.draw.line(
        background,
        (255, 255, 255),
        (160, MAX_Y_PLOT + 155),
        (175, MAX_Y_PLOT + 55)
    )


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
    
    
    draw_network(ox, oy, background)

    window_surface.blit(background, (0, 0))
    manager.draw_ui(window_surface)

    pygame.display.update()