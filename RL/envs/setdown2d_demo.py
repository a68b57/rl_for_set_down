import spec_tools.spec_tools as st
from pymunk.vec2d import Vec2d
import numpy as np
import pygame
import matplotlib.pyplot as plt

from game_2d_physics import game2d


fps = 60   # this many frames per second for drawing
scale = 5  # drawing scale
rtf = 5    # time scale

engine = game2d()

# set some of the properities
engine.wave_spectrum = st.Spectrum.from_synthetic(Hs=2, coming_from=0, omega=np.arange(0,4,0.01), Tp=8,gamma=2.0,
                                           spreading_type=None)
engine.magic_pitch_factor = 0.2 # <1 means lower pitch

# dt_inner of 0.002 seems to give consistent results
#
# rtf / (fps * ninner) <= 0.002
# so
# r_inner = rtf / (dt_inner_target * fps)
#
dt_inner_target = 0.004
engine.n_inner = int(np.ceil(rtf / (dt_inner_target * fps)))
print('Using {} inner iterations per time-step'.format(engine.n_inner))

engine.setup()
engine.prep_new_run()

pygame.init()
clock = pygame.time.Clock()
pygame.key.set_repeat(True)
display = pygame.display.set_mode((800, 600))
ball_img = pygame.image.load('ball.png')
background = pygame.image.load('background.png')

def plot_ball(vec):
    display.blit(ball_img, (vec[0] - ball_img.get_width()/2,vec[1] - ball_img.get_height()/2))

while True:

    engine.prep_new_run()
    engine.is_done = False

    sim_start = rtf*pygame.time.get_ticks() / 1000

    while not engine.is_done:

        action = 'hold'  # hold

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                engine.is_done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 'up'
                if event.key == pygame.K_DOWN:
                    action = 'down'
                if event.key == pygame.K_LEFT:
                    action = 'left'
                if event.key == pygame.K_RIGHT:
                    action = 'right'

                if event.key == 113:
                    print('quit')
                    engine.is_done = True
    #
    #     # game physics
        t_simulation = rtf*pygame.time.get_ticks() / 1000 # convert to seconds
        t_simulation = t_simulation - sim_start
        engine.step(t_simulation,rtf/fps, action)



    #     # Update graphics
    #
        def p2d(p):  # physics to display
            p = scale * p
            p[0] = p[0] + 400
            return p

        def p2dl(pl):
            r = []
            for p in pl:
                r.append(p2d(p))
            return r

        display.blit(background, (0, 0))
        plot_ball(p2d(engine.hook.position))

        global_poi_position = engine.load.local_to_world(engine.poi)
        pygame.draw.aaline(display, [255, 0, 0], p2d(engine.hook.position), p2d(global_poi_position), 3)
    #
    #     # draw barge
        p1 = engine.barge.local_to_world(engine.barge_lower_left)
        p2 = engine.barge.local_to_world(engine.barge_lower_right)
        p3 = engine.barge.local_to_world(engine.barge_upper_right)
        p4 = engine.barge.local_to_world(engine.barge_upper_left)

        if engine.has_barge_contact:
            col = [0,254,0]
        else:
            col = [0,0,0]

        # pygame.draw.aalines(display, col, True, p2dl([p1, p2, p3, p4]), 3)
        pygame.draw.polygon(display, col, p2dl([p1, p2, p3, p4]), 0)

        # draw load
        p1 = engine.load.local_to_world(engine.load_lower_left)
        p2 = engine.load.local_to_world(engine.load_lower_right)
        p3 = engine.load.local_to_world(engine.load_upper_right)
        p4 = engine.load.local_to_world(engine.load_upper_left)
        pygame.draw.polygon(display, [30, 0, 0], p2dl([p1, p2, p3, p4]), 0)
        pygame.draw.aalines(display, [0, 0, 0], True, p2dl([p1, p2, p3, p4]), 3)

        # draw engine.bumper
        p1 = engine.barge.local_to_world(engine.bumper_lower)
        p2 = engine.barge.local_to_world(engine.bumper_upper)

        if engine.has_bumper_contact:
            col = [0,254,0]
        else:
            col = [0,0,0]
        pygame.draw.aaline(display, col, p2d(p1), p2d(p2), 4)

        # draw rigging
        p1 = engine.load.local_to_world(engine.load_upper_right)
        p2 = engine.load.local_to_world(engine.poi)
        p3 = engine.load.local_to_world(engine.load_upper_left)

        pygame.draw.aalines(display, [0, 0, 0], True, p2dl([p1,p2,p3]), 3)

        # draw wave
        points = []

        for i, r in enumerate(engine.wave_elevation):
            e = np.interp(t_simulation, engine.motions_t, r)
            point = Vec2d(engine.wave_location[i],engine.water_level-e)
            points.append(point)

        pygame.draw.aalines(display, [0, 200, 254], False, p2dl(points), 4)

        pygame.display.update()
        clock.tick(fps)

    plt.plot(engine.bumper_impulse,label = 'bumper (x and y)')
    plt.plot(engine.barge_impulse,label = 'deck (x and y)')
    plt.legend()
    plt.show()