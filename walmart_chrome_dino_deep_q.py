"""
Chrome Dino RL - Deep Q-Network (DQN), single file, PyTorch.

Usage:
  python dino_dqn.py --train 3000          # headless train, saves dqn.pt
  python dino_dqn.py --play                # loads dqn.pt, renders with pygame
  python dino_dqn.py --train 3000 --play

Deps: pip install torch pygame numpy
"""

import argparse, random, os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame


# environment information
W, H = 600, 150
GROUND_Y = 120
GRAVITY = 1.2
JUMP_V = -15
DINO_X = 50
DINO_SIZE = 20
SPEED = 7
STATE_DIM = 6   # see get_state() below
N_ACTIONS = 2   # 0 = do nothing, 1 = jump

class DinoEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y = GROUND_Y
        self.vy = 0
        self.jumping = False
        self.obstacles = []
        self.spawn_timer = random.randint(30, 60)
        self.score = 0
        return self._state()

    def _spawn(self):
        h = random.choice([15, 20, 30])
        w = random.choice([10, 15, 20])
        self.obstacles.append([W, w, h])

    def step(self, action):
        if action == 1 and not self.jumping:
            self.vy = JUMP_V
            self.jumping = True

        self.y += self.vy
        self.vy += GRAVITY
        if self.y >= GROUND_Y:
            self.y, self.vy, self.jumping = GROUND_Y, 0, False

        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn()
            self.spawn_timer = random.randint(35, 70)

        for o in self.obstacles:
            o[0] -= SPEED
        self.obstacles = [o for o in self.obstacles if o[0] + o[1] > 0]

        done = False
        dino_top = self.y - DINO_SIZE
        for ox, ow, oh in self.obstacles:
            obs_top = GROUND_Y - oh
            if (DINO_X + DINO_SIZE > ox and DINO_X < ox + ow and
                    dino_top < GROUND_Y and dino_top + DINO_SIZE > obs_top):
                done = True
                break

        self.score += 1
        reward = 1.0 if not done else -100.0
        return self._state(), reward, done

    def _state(self):
        """
        6 continuous features, all roughly normalized to [-1, 1] / [0, 1]:
          0: dx to nearest obstacle ahead   (0 = right on top of dino, 1 = far away)
          1: nearest obstacle height        (0..1)
          2: nearest obstacle width         (0..1)
          3: dx to 2nd obstacle ahead       (1.0 if none queued -> "far")
          4: dino height above ground       (0 = on ground, 1 = top of jump)
          5: dino vertical velocity         (-1 falling fast .. 1 rising fast)
        """
        ahead = [o for o in self.obstacles if o[0] + o[1] >= DINO_X]
        if len(ahead) >= 1:
            o = ahead[0]
            dx0 = min(1.0, max(0.0, (o[0] - DINO_X) / W))
            h0 = min(1.0, o[2] / 40)
            w0 = min(1.0, o[1] / 30)
        else:
            dx0, h0, w0 = 1.0, 0.0, 0.0
        if len(ahead) >= 2:
            dx1 = min(1.0, max(0.0, (ahead[1][0] - DINO_X) / W))
        else:
            dx1 = 1.0

        y_norm = min(1.0, max(0.0, (GROUND_Y - self.y) / 45))
        vy_norm = max(-1.0, min(1.0, self.vy / 15))

        return np.array([dx0, h0, w0, dx1, y_norm, vy_norm], dtype=np.float32)


# ---------------- DQN ----------------
class QNet(nn.Module):
    def __init__(self, in_dim=STATE_DIM, out_dim=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim),   # raw Q-values, no activation
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap=50_000):
        self.buf = deque(maxlen=cap)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s)),
                torch.tensor(a, dtype=torch.int64).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2)),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


def train(episodes, path="dqn.pt"):
    device = torch.device("cpu")
    policy = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=1e-3)
    buf = ReplayBuffer()

    gamma = 0.95
    # epsilon, epsilon decay (to slowly reduce exploration), eps_min (can't go under it)
    eps, eps_min, eps_decay = 1.0, 0.02, 0.998
    batch_size = 64
    target_update_every = 500
    step_count = 0
    best = 0

    env = DinoEnv()
    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(s).unsqueeze(0))
                    a = int(q.argmax(dim=1).item())

            s2, r, done = env.step(a)
            buf.push(s, a, r, s2, float(done))
            s = s2
            step_count += 1

            if len(buf) >= batch_size:
                # batch states, batch actions, batch rewards, batch_next_states, batch_dones (fancy word for terminal states)
                bs, ba, br, bs2, bd = buf.sample(batch_size)
                q_pred = policy(bs).gather(1, ba)             # Q(s,a) for actions taken
                with torch.no_grad():
                    q_next = target(bs2).max(1, keepdim=True)[0]
                    q_target = br + gamma * q_next * (1 - bd)
                loss = nn.functional.mse_loss(q_pred, q_target)
                opt.zero_grad(); loss.backward(); opt.step()

            if step_count % target_update_every == 0:
                target.load_state_dict(policy.state_dict())

        eps = max(eps_min, eps * eps_decay)
        best = max(best, env.score)
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1}/{episodes}  eps={eps:.3f}  last_score={env.score}  best={best}")

    torch.save(policy.state_dict(), path)
    print(f"saved {path}, best score {best}")


def play(path="dqn.pt"):
    policy = QNet()
    if os.path.exists(path):
        policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()

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
        with torch.no_grad():
            q = policy(torch.tensor(s).unsqueeze(0))
            a = int(q.argmax(dim=1).item())
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
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--path", default="dqn.pt")
    args = ap.parse_args()

    if args.train:
        train(args.train, args.path)
    if args.play:
        play(args.path)
    if not args.train and not args.play:
        print("nothing to do — pass --train N and/or --play")