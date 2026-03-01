"""
Dynamic Pathfinding Agent
NUCES Chiniot-Faisalabad Campus
Implements GBFS and A* with Manhattan & Euclidean heuristics
"""

import pygame
import heapq
import random
import time
import math

# ── Constants ──────────────────────────────────────────────────────────────────
CELL = 28          # pixel size of each cell
PANEL_W = 260      # right-side control panel width

# Colours
C_BG        = (18,  18,  26)
C_GRID      = (35,  35,  50)
C_EMPTY     = (28,  28,  40)
C_WALL      = (60,  60,  80)
C_START     = (50, 200, 100)
C_GOAL      = (220, 70,  70)
C_FRONTIER  = (240, 200,  30)   # yellow
C_VISITED   = (70,  100, 200)   # blue
C_PATH      = (50,  220, 120)   # green
C_AGENT     = (255, 255, 255)
C_PANEL     = (22,  22,  34)
C_TEXT      = (200, 200, 220)
C_ACCENT    = (100, 180, 255)
C_BTN       = (40,  40,  60)
C_BTN_HOV   = (60,  60,  90)
C_BTN_ACT   = (70, 120, 200)

pygame.init()
FONT_SM  = pygame.font.SysFont("consolas", 12)
FONT_MED = pygame.font.SysFont("consolas", 14, bold=True)
FONT_LG  = pygame.font.SysFont("consolas", 16, bold=True)


# ── Heuristics ─────────────────────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


# ── Search Algorithms ──────────────────────────────────────────────────────────
def get_neighbors(pos, grid, rows, cols):
    r, c = pos
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def gbfs(grid, start, goal, rows, cols, heuristic_fn):
    """Greedy Best-First Search: f(n) = h(n)"""
    frontier = []
    heapq.heappush(frontier, (heuristic_fn(start, goal), start))
    came_from = {}
    visited = set()
    frontier_set = {start}
    nodes_visited = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        frontier_set.discard(current)

        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1

        if current == goal:
            return reconstruct_path(came_from, goal), visited, nodes_visited

        for nb in get_neighbors(current, grid, rows, cols):
            if nb not in visited:
                came_from[nb] = current
                heapq.heappush(frontier, (heuristic_fn(nb, goal), nb))
                frontier_set.add(nb)

    return None, visited, nodes_visited

def astar(grid, start, goal, rows, cols, heuristic_fn):
    """A* Search: f(n) = g(n) + h(n)"""
    frontier = []
    heapq.heappush(frontier, (heuristic_fn(start, goal), 0, start))
    came_from = {}
    g_cost = {start: 0}
    visited = set()
    nodes_visited = 0

    while frontier:
        f, g, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1

        if current == goal:
            return reconstruct_path(came_from, goal), visited, nodes_visited

        for nb in get_neighbors(current, grid, rows, cols):
            new_g = g_cost[current] + 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                f_val = new_g + heuristic_fn(nb, goal)
                heapq.heappush(frontier, (f_val, new_g, nb))
                came_from[nb] = current

    return None, visited, nodes_visited


# ── Button Helper ──────────────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, active=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.active = active

    def draw(self, surface, mouse_pos):
        hover = self.rect.collidepoint(mouse_pos)
        if self.active:
            col = C_BTN_ACT
        elif hover:
            col = C_BTN_HOV
        else:
            col = C_BTN

        pygame.draw.rect(surface, col, self.rect, border_radius=6)
        pygame.draw.rect(surface, C_ACCENT if self.active else C_GRID, self.rect, 1, border_radius=6)
        txt = FONT_SM.render(self.label, True, C_TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


# ── Main Application ───────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.ROWS = 20
        self.COLS = 25
        self.grid = [[0]*self.COLS for _ in range(self.ROWS)]

        self.start = (1, 1)
        self.goal  = (self.ROWS-2, self.COLS-2)
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]]   = 0

        # State
        self.path         = []
        self.visited      = set()
        self.frontier     = set()
        self.agent_idx    = 0
        self.running_anim = False
        self.dynamic_mode = False
        self.placing      = "wall"    # wall | start | goal

        # Algo / heuristic selection
        self.algo = "astar"           # astar | gbfs
        self.heur = "manhattan"       # manhattan | euclidean

        # Metrics
        self.nodes_visited = 0
        self.path_cost     = 0
        self.exec_time     = 0.0
        self.status_msg    = "Ready"

        # Window
        grid_w = self.COLS * CELL
        grid_h = self.ROWS * CELL
        self.WIN_W = grid_w + PANEL_W
        self.WIN_H = max(grid_h, 600)
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()

        self._build_ui()

    def _build_ui(self):
        px = self.COLS * CELL + 10
        y  = 10
        bw, bh = PANEL_W - 20, 28

        def btn(label, active=False):
            nonlocal y
            b = Button((px, y, bw, bh), label, active)
            y += bh + 6
            return b

        def gap(n=8):
            nonlocal y; y += n

        # Algorithm
        self.btn_astar = btn("A* Search",   self.algo=="astar")
        self.btn_gbfs  = btn("Greedy BFS",  self.algo=="gbfs")
        gap()
        # Heuristic
        self.btn_manh  = btn("Manhattan",   self.heur=="manhattan")
        self.btn_eucl  = btn("Euclidean",   self.heur=="euclidean")
        gap()
        # Actions
        self.btn_run   = btn("▶  Run Search")
        self.btn_step  = btn("⏭  Step")
        self.btn_clear = btn("✕  Clear Path")
        gap()
        # Map tools
        self.btn_rand  = btn("⊞  Random Map")
        self.btn_reset = btn("↺  Reset Grid")
        gap()
        # Place modes
        self.btn_wall  = btn("✏  Place Wall",  self.placing=="wall")
        self.btn_start = btn("S  Move Start",  self.placing=="start")
        self.btn_goal  = btn("G  Move Goal",   self.placing=="goal")
        gap()
        # Dynamic
        self.btn_dyn   = btn("⚡ Dynamic Mode", self.dynamic_mode)

        self.all_buttons = [
            self.btn_astar, self.btn_gbfs,
            self.btn_manh,  self.btn_eucl,
            self.btn_run,   self.btn_step,  self.btn_clear,
            self.btn_rand,  self.btn_reset,
            self.btn_wall,  self.btn_start, self.btn_goal,
            self.btn_dyn
        ]

    def _heuristic(self):
        return manhattan if self.heur == "manhattan" else euclidean

    def _algo_fn(self):
        return astar if self.algo == "astar" else gbfs

    def run_search(self):
        t0 = time.perf_counter()
        fn = self._algo_fn()
        path, vis, nv = fn(self.grid, self.start, self.goal,
                           self.ROWS, self.COLS, self._heuristic())
        self.exec_time = (time.perf_counter() - t0) * 1000

        self.visited = vis
        self.nodes_visited = nv
        if path:
            self.path = path
            self.path_cost = len(path)
            self.agent_idx = 0
            self.running_anim = True
            self.status_msg = "Path found!"
        else:
            self.path = []
            self.path_cost = 0
            self.status_msg = "No path found!"

    def clear_path(self):
        self.path = []
        self.visited = set()
        self.frontier = set()
        self.agent_idx = 0
        self.running_anim = False
        self.nodes_visited = 0
        self.path_cost = 0
        self.exec_time = 0.0
        self.status_msg = "Cleared"

    def random_map(self, density=0.28):
        self.clear_path()
        self.grid = [[0]*self.COLS for _ in range(self.ROWS)]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if (r, c) in (self.start, self.goal):
                    continue
                self.grid[r][c] = 1 if random.random() < density else 0

    def reset_grid(self):
        self.clear_path()
        self.grid = [[0]*self.COLS for _ in range(self.ROWS)]

    def _cell_at(self, mx, my):
        c = mx // CELL
        r = my // CELL
        if 0 <= r < self.ROWS and 0 <= c < self.COLS:
            return r, c
        return None

    # Dynamic obstacle spawning + re-planning
    def _dynamic_step(self):
        if not self.running_anim or not self.path:
            return
        # small chance to spawn a wall on a random empty cell not on path
        if random.random() < 0.08:
            r = random.randint(0, self.ROWS-1)
            c = random.randint(0, self.COLS-1)
            pos = (r, c)
            if (pos not in (self.start, self.goal) and
                    self.grid[r][c] == 0):
                self.grid[r][c] = 1
                # check if obstacle is on current path
                remaining = self.path[self.agent_idx:]
                if pos in remaining:
                    # re-plan from current agent position
                    current_pos = self.path[self.agent_idx-1] if self.agent_idx > 0 else self.start
                    fn = self._algo_fn()
                    new_path, vis, nv = fn(self.grid, current_pos, self.goal,
                                          self.ROWS, self.COLS, self._heuristic())
                    self.visited |= vis
                    self.nodes_visited += nv
                    if new_path:
                        self.path = new_path
                        self.agent_idx = 0
                        self.status_msg = "Re-planned!"
                    else:
                        self.running_anim = False
                        self.status_msg = "Path blocked!"

    def _advance_agent(self):
        if self.running_anim and self.path:
            if self.agent_idx < len(self.path):
                self.agent_idx += 1
            if self.agent_idx >= len(self.path):
                self.running_anim = False
                self.status_msg = "Goal reached!"

    def _draw_grid(self):
        path_set = set(self.path)
        for r in range(self.ROWS):
            for c in range(self.COLS):
                x, y = c*CELL, r*CELL
                rect = (x, y, CELL-1, CELL-1)
                pos = (r, c)

                if self.grid[r][c] == 1:
                    col = C_WALL
                elif pos == self.start:
                    col = C_START
                elif pos == self.goal:
                    col = C_GOAL
                elif pos in path_set:
                    col = C_PATH
                elif pos in self.visited:
                    col = C_VISITED
                else:
                    col = C_EMPTY

                pygame.draw.rect(self.screen, col, rect, border_radius=3)

                # grid lines
                pygame.draw.rect(self.screen, C_GRID, rect, 1, border_radius=3)

        # Draw agent
        if self.path and 0 < self.agent_idx <= len(self.path):
            ar, ac = self.path[self.agent_idx-1]
            cx = ac*CELL + CELL//2
            cy = ar*CELL + CELL//2
            pygame.draw.circle(self.screen, C_AGENT, (cx, cy), CELL//2 - 3)

    def _draw_panel(self):
        px = self.COLS * CELL
        panel_rect = pygame.Rect(px, 0, PANEL_W, self.WIN_H)
        pygame.draw.rect(self.screen, C_PANEL, panel_rect)
        pygame.draw.line(self.screen, C_GRID, (px, 0), (px, self.WIN_H), 2)

        mouse = pygame.mouse.get_pos()

        # Title
        t = FONT_LG.render("PATHFINDER", True, C_ACCENT)
        self.screen.blit(t, (px+10, 8))
        # section labels
        sections = {
            self.btn_astar: "ALGORITHM",
            self.btn_manh:  "HEURISTIC",
            self.btn_run:   "CONTROLS",
            self.btn_rand:  "MAP",
            self.btn_wall:  "DRAW MODE",
            self.btn_dyn:   "DYNAMIC",
        }
        offset_map = {}
        for b in self.all_buttons:
            if b in sections:
                offset_map[b] = sections[b]

        for b in self.all_buttons:
            if b in offset_map:
                lbl = FONT_SM.render(offset_map[b], True, (80, 80, 110))
                self.screen.blit(lbl, (px+10, b.rect.y - 14))
            b.draw(self.screen, mouse)

        # Metrics
        my = self.btn_dyn.rect.bottom + 20
        pygame.draw.line(self.screen, C_GRID, (px+10, my), (px+PANEL_W-10, my))
        my += 8
        metrics = [
            ("STATUS",        self.status_msg),
            ("ALGORITHM",     self.algo.upper()),
            ("HEURISTIC",     self.heur.upper()),
            ("NODES VISITED", str(self.nodes_visited)),
            ("PATH COST",     str(self.path_cost)),
            ("EXEC TIME",     f"{self.exec_time:.2f} ms"),
        ]
        for label, val in metrics:
            lt = FONT_SM.render(label, True, (80, 80, 110))
            vt = FONT_MED.render(val,  True, C_TEXT)
            self.screen.blit(lt, (px+10, my))
            my += 14
            self.screen.blit(vt, (px+10, my))
            my += 20

        # Legend
        my += 4
        pygame.draw.line(self.screen, C_GRID, (px+10, my), (px+PANEL_W-10, my))
        my += 8
        legend = [
            (C_START,    "Start"),
            (C_GOAL,     "Goal"),
            (C_WALL,     "Wall"),
            (C_VISITED,  "Visited"),
            (C_PATH,     "Path"),
        ]
        for col, lbl in legend:
            pygame.draw.rect(self.screen, col, (px+10, my, 14, 14), border_radius=3)
            t = FONT_SM.render(lbl, True, C_TEXT)
            self.screen.blit(t, (px+28, my))
            my += 18

        # Keybinds hint
        my += 4
        hints = ["ENTER=Run  C=Clear", "R=Random  DEL=Reset", "D=Dynamic Mode"]
        for h in hints:
            t = FONT_SM.render(h, True, (70, 70, 95))
            self.screen.blit(t, (px+10, my))
            my += 14

    def _handle_click(self, pos, erase=False):
        cell = self._cell_at(*pos)
        if cell is None:
            return
        r, c = cell
        if self.placing == "wall":
            if cell not in (self.start, self.goal):
                self.grid[r][c] = 0 if erase else 1
        elif self.placing == "start":
            self.start = cell
            self.grid[r][c] = 0
        elif self.placing == "goal":
            self.goal = cell
            self.grid[r][c] = 0

    def _update_btn_states(self):
        self.btn_astar.active = self.algo == "astar"
        self.btn_gbfs.active  = self.algo == "gbfs"
        self.btn_manh.active  = self.heur == "manhattan"
        self.btn_eucl.active  = self.heur == "euclidean"
        self.btn_wall.active  = self.placing == "wall"
        self.btn_start.active = self.placing == "start"
        self.btn_goal.active  = self.placing == "goal"
        self.btn_dyn.active   = self.dynamic_mode

    def run(self):
        ANIM_DELAY = 80   # ms per agent step
        DYN_DELAY  = 300  # ms per dynamic step
        last_anim  = pygame.time.get_ticks()
        last_dyn   = pygame.time.get_ticks()
        mouse_held = False
        erase_mode = False

        while True:
            now = pygame.time.get_ticks()

            # Agent animation
            if self.running_anim and now - last_anim > ANIM_DELAY:
                self._advance_agent()
                last_anim = now

            # Dynamic obstacle spawning
            if self.dynamic_mode and self.running_anim and now - last_dyn > DYN_DELAY:
                self._dynamic_step()
                last_dyn = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.clear_path(); self.run_search()
                    elif event.key == pygame.K_c:
                        self.clear_path()
                    elif event.key == pygame.K_r:
                        self.random_map()
                    elif event.key == pygame.K_DELETE:
                        self.reset_grid()
                    elif event.key == pygame.K_d:
                        self.dynamic_mode = not self.dynamic_mode

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if mx < self.COLS * CELL:   # on grid
                        mouse_held = True
                        erase_mode = (event.button == 3)  # right-click = erase
                        self._handle_click((mx, my), erase=erase_mode)
                    else:                       # on panel
                        if self.btn_astar.clicked(event.pos):
                            self.algo = "astar"
                        elif self.btn_gbfs.clicked(event.pos):
                            self.algo = "gbfs"
                        elif self.btn_manh.clicked(event.pos):
                            self.heur = "manhattan"
                        elif self.btn_eucl.clicked(event.pos):
                            self.heur = "euclidean"
                        elif self.btn_run.clicked(event.pos):
                            self.clear_path(); self.run_search()
                        elif self.btn_step.clicked(event.pos):
                            if not self.path:
                                self.run_search()
                                self.running_anim = False
                            else:
                                self._advance_agent()
                        elif self.btn_clear.clicked(event.pos):
                            self.clear_path()
                        elif self.btn_rand.clicked(event.pos):
                            self.random_map()
                        elif self.btn_reset.clicked(event.pos):
                            self.reset_grid()
                        elif self.btn_wall.clicked(event.pos):
                            self.placing = "wall"
                        elif self.btn_start.clicked(event.pos):
                            self.placing = "start"
                        elif self.btn_goal.clicked(event.pos):
                            self.placing = "goal"
                        elif self.btn_dyn.clicked(event.pos):
                            self.dynamic_mode = not self.dynamic_mode

                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_held = False

                elif event.type == pygame.MOUSEMOTION:
                    if mouse_held:
                        mx, my = event.pos
                        if mx < self.COLS * CELL:
                            self._handle_click((mx, my), erase=erase_mode)

            self._update_btn_states()

            # Draw
            self.screen.fill(C_BG)
            self._draw_grid()
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    App().run()