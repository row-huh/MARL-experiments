"""
Chrome Dino RL - tabular Q-learning, single file.

Usage:
  python dino_rl.py --train 20000        # headless train, saves qtable.npy
  python dino_rl.py --play               # loads qtable.npy, renders with pygame
  python dino_rl.py --train 20000 --play # train then watch it play

Deps: pip install pygame numpy
"""

import argparse, random, pickle, os
import numpy as np


W, H = 600, 150
GROUND_Y = 120
GRAVITY = 1.2
JUMP_V = -15
DINO_X = 50
DINO_SIZE = 20
SPEED = 7


class DinoEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y = GROUND_Y
        self.vy = 0
        self.jumping = False
        self.obstacles = []  # list of [x, w, h]
        self.spawn_timer = random.randint(30, 60)
        self.t = 0
        self.score = 0
        return self._state()

    def _spawn(self):
        h = random.choice([15, 20, 30])
        w = random.choice([10, 15, 20])
        self.obstacles.append([W, w, h])

    def step(self, action):
        # action: 0 = nothing, 1 = jump
        if action == 1 and not self.jumping:
            self.vy = JUMP_V
            self.jumping = True

        self.y += self.vy
        self.vy += GRAVITY
        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            self.vy = 0
            self.jumping = False

        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn()
            self.spawn_timer = random.randint(35, 70)

        done = False
        for o in self.obstacles:
            o[0] -= SPEED
        self.obstacles = [o for o in self.obstacles if o[0] + o[1] > 0]

        # collision check (dino is a box at DINO_X, self.y up to DINO_SIZE tall)
        dino_top = self.y - DINO_SIZE
        for ox, ow, oh in self.obstacles:
            obs_top = GROUND_Y - oh
            if (DINO_X + DINO_SIZE > ox and DINO_X < ox + ow and
                    dino_top < GROUND_Y and dino_top + DINO_SIZE > obs_top):
                done = True
                break

        self.t += 1
        self.score += 1
        reward = 1.0 if not done else -100.0
        return self._state(), reward, done

    def _state(self):
        # find nearest obstacle ahead
        nearest = None
        for o in self.obstacles:
            if o[0] + o[1] >= DINO_X:
                nearest = o
                break
        if nearest is None:
            dist_b, h_b = 9, 0
        else:
            dist = max(0, nearest[0] - DINO_X)
            dist_b = min(9, dist // 40)
            h_b = min(3, nearest[2] // 10)
        y_b = min(4, max(0, int((GROUND_Y - self.y) // 10)))
        vy_b = 0 if self.vy < -1 else (1 if self.vy > 1 else 2)
        return (dist_b, h_b, y_b, vy_b)


# ---------------- Q-learning ----------------
ACTIONS = [0, 1]
QFILE = "qtable.npy"

def train(episodes, qfile=QFILE):
    Q = {}
    alpha, gamma = 0.2, 0.95
    eps, eps_min, eps_decay = 1.0, 0.02, 0.9995

    def get(s):
        if s not in Q:
            Q[s] = np.zeros(len(ACTIONS))
        return Q[s]

    env = DinoEnv()
    best = 0
    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = int(np.argmax(get(s)))
            s2, r, done = env.step(a)
            get(s)[a] += alpha * (r + gamma * np.max(get(s2)) - get(s)[a])
            s = s2
        eps = max(eps_min, eps * eps_decay)
        best = max(best, env.score)
        if (ep + 1) % 1000 == 0:
            print(f"ep {ep+1}/{episodes}  eps={eps:.3f}  last_score={env.score}  best={best}")

    with open(qfile, "wb") as f:
        pickle.dump(Q, f)
    print(f"saved {qfile}, {len(Q)} states, best score {best}")
    return Q


def load_q(qfile=QFILE):
    with open(qfile, "rb") as f:
        return pickle.load(f)


def play(qfile=QFILE):
    import pygame
    Q = load_q(qfile) if os.path.exists(qfile) else {}
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    env = DinoEnv()
    s = env.reset()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        a = int(np.argmax(Q[s])) if s in Q else 0
        s, r, done = env.step(a)
        if done:
            s = env.reset()

        screen.fill((255, 255, 255))
        pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (W, GROUND_Y), 2)
        pygame.draw.rect(screen, (50, 50, 50),
                          (DINO_X, env.y - DINO_SIZE, DINO_SIZE, DINO_SIZE))
        for ox, ow, oh in env.obstacles:
            pygame.draw.rect(screen, (200, 30, 30), (ox, GROUND_Y - oh, ow, oh))
        screen.blit(font.render(f"score {env.score}", True, (0, 0, 0)), (10, 10))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0, help="episodes to train")
    ap.add_argument("--play", action="store_true", help="render + play with saved qtable")
    ap.add_argument("--qfile", default=QFILE)
    args = ap.parse_args()

    if args.train:
        train(args.train, args.qfile)
    if args.play:
        play(args.qfile)
    if not args.train and not args.play:
        print("nothing to do — pass --train N and/or --play")