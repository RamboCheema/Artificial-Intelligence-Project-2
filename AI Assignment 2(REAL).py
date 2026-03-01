import pygame
import heapq
import random
import time
import math

CELL    = 24
PANEL_W = 280

C_BG       = (18,  18,  26)
C_GRID     = (35,  35,  50)
C_EMPTY    = (28,  28,  40)
C_WALL     = (60,  60,  80)
C_START    = (50,  200, 100)
C_GOAL     = (220, 70,  70)
C_VISITED  = (70,  100, 200)
C_PATH     = (50,  220, 120)
C_AGENT    = (255, 255, 255)
C_PANEL    = (22,  22,  34)
C_PANEL2   = (28,  28,  42)
C_TEXT     = (200, 200, 220)
C_LABEL    = (90,  90,  120)
C_ACCENT   = (100, 180, 255)
C_BTN      = (40,  40,  60)
C_BTN_HOV  = (60,  60,  90)
C_BTN_ACT  = (70,  120, 200)
C_GREEN    = (50,  220, 120)
C_YELLOW   = (240, 200, 30)
C_RED      = (220, 70,  70)

pygame.init()
FONT_XS  = pygame.font.SysFont("consolas", 11)
FONT_SM  = pygame.font.SysFont("consolas", 12)
FONT_MED = pygame.font.SysFont("consolas", 13, bold=True)
FONT_LG  = pygame.font.SysFont("consolas", 15, bold=True)
FONT_XL  = pygame.font.SysFont("consolas", 22, bold=True)


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def get_neighbors(pos, grid, rows, cols):
    r, c = pos
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            result.append((nr, nc))
    return result

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def gbfs(grid, start, goal, rows, cols, hfn):
    frontier = []
    heapq.heappush(frontier, (hfn(start, goal), start))
    came_from = {}
    visited = set()
    count = 0
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur in visited:
            continue
        visited.add(cur)
        count += 1
        if cur == goal:
            return reconstruct_path(came_from, goal), visited, count
        for nb in get_neighbors(cur, grid, rows, cols):
            if nb not in visited:
                came_from[nb] = cur
                heapq.heappush(frontier, (hfn(nb, goal), nb))
    return None, visited, count

def astar(grid, start, goal, rows, cols, hfn):
    frontier = []
    heapq.heappush(frontier, (hfn(start, goal), 0, start))
    came_from = {}
    g_cost = {start: 0}
    visited = set()
    count = 0
    while frontier:
        f, g, cur = heapq.heappop(frontier)
        if cur in visited:
            continue
        visited.add(cur)
        count += 1
        if cur == goal:
            return reconstruct_path(came_from, goal), visited, count
        for nb in get_neighbors(cur, grid, rows, cols):
            new_g = g_cost[cur] + 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                heapq.heappush(frontier, (new_g + hfn(nb, goal), new_g, nb))
                came_from[nb] = cur
    return None, visited, count


class Button:
    def __init__(self, rect, label, active=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.active = active

    def draw(self, surface, mouse_pos):
        hover = self.rect.collidepoint(mouse_pos)
        col = C_BTN_ACT if self.active else (C_BTN_HOV if hover else C_BTN)
        pygame.draw.rect(surface, col, self.rect, border_radius=5)
        border_col = C_ACCENT if self.active else C_GRID
        pygame.draw.rect(surface, border_col, self.rect, 1, border_radius=5)
        txt = FONT_SM.render(self.label, True, C_TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class App:
    def __init__(self):
        self.ROWS = 22
        self.COLS = 28
        self.grid = [[0]*self.COLS for _ in range(self.ROWS)]

        self.start = (1, 1)
        self.goal  = (self.ROWS-2, self.COLS-2)

        self.path         = []
        self.visited      = set()
        self.agent_idx    = 0
        self.running_anim = False
        self.dynamic_mode = False
        self.placing      = "wall"
        self.algo         = "astar"
        self.heur         = "manhattan"

        self.nodes_visited = 0
        self.path_cost     = 0
        self.exec_time     = 0.0
        self.status_msg    = "Ready"
        self.status_col    = C_TEXT

        grid_w = self.COLS * CELL
        grid_h = self.ROWS * CELL
        self.WIN_W = grid_w + PANEL_W
        self.WIN_H = max(grid_h, 780)
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()
        self._build_ui()

    def _build_ui(self):
        px = self.COLS * CELL + 10
        y  = 36
        bw, bh = PANEL_W - 20, 26

        def btn(label, active=False):
            nonlocal y
            b = Button((px, y, bw, bh), label, active)
            y += bh + 5
            return b

        def gap(n=10):
            nonlocal y; y += n

        self.btn_astar = btn("A* Search",      self.algo=="astar")
        self.btn_gbfs  = btn("Greedy BFS",     self.algo=="gbfs")
        gap()
        self.btn_manh  = btn("Manhattan",      self.heur=="manhattan")
        self.btn_eucl  = btn("Euclidean",      self.heur=="euclidean")
        gap()
        self.btn_run   = btn("▶  Run Search")
        self.btn_step  = btn("⏭  Step")
        self.btn_clear = btn("✕  Clear Path")
        gap()
        self.btn_rand  = btn("⊞  Random Map")
        self.btn_reset = btn("↺  Reset Grid")
        gap()
        self.btn_wall  = btn("✏  Place Wall",  self.placing=="wall")
        self.btn_start = btn("S  Move Start",  self.placing=="start")
        self.btn_goal  = btn("G  Move Goal",   self.placing=="goal")
        gap()
        self.btn_dyn   = btn("⚡ Dynamic Mode", self.dynamic_mode)

        self.all_buttons = [
            self.btn_astar, self.btn_gbfs,
            self.btn_manh,  self.btn_eucl,
            self.btn_run,   self.btn_step, self.btn_clear,
            self.btn_rand,  self.btn_reset,
            self.btn_wall,  self.btn_start, self.btn_goal,
            self.btn_dyn
        ]

        self.metrics_y = self.btn_dyn.rect.bottom + 14

    def _heuristic(self):
        return manhattan if self.heur == "manhattan" else euclidean

    def _algo_fn(self):
        return astar if self.algo == "astar" else gbfs

    def run_search(self):
        t0 = time.perf_counter()
        path, vis, nv = self._algo_fn()(
            self.grid, self.start, self.goal,
            self.ROWS, self.COLS, self._heuristic()
        )
        self.exec_time     = (time.perf_counter() - t0) * 1000
        self.visited       = vis
        self.nodes_visited = nv
        if path:
            self.path         = path
            self.path_cost    = len(path)
            self.agent_idx    = 0
            self.running_anim = True
            self.status_msg   = "Path found!"
            self.status_col   = C_GREEN
        else:
            self.path       = []
            self.path_cost  = 0
            self.status_msg = "No path found!"
            self.status_col = C_RED

    def clear_path(self):
        self.path          = []
        self.visited       = set()
        self.agent_idx     = 0
        self.running_anim  = False
        self.nodes_visited = 0
        self.path_cost     = 0
        self.exec_time     = 0.0
        self.status_msg    = "Cleared"
        self.status_col    = C_TEXT

    def random_map(self, density=0.28):
        self.clear_path()
        self.grid = [[0]*self.COLS for _ in range(self.ROWS)]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if (r, c) not in (self.start, self.goal):
                    self.grid[r][c] = 1 if random.random() < density else 0

    def reset_grid(self):
        self.clear_path()
        self.grid = [[0]*self.COLS for _ in range(self.ROWS)]

    def _cell_at(self, mx, my):
        c, r = mx // CELL, my // CELL
        if 0 <= r < self.ROWS and 0 <= c < self.COLS:
            return r, c
        return None

    def _dynamic_step(self):
        if not self.running_anim or not self.path:
            return
        if random.random() < 0.08:
            r = random.randint(0, self.ROWS-1)
            c = random.randint(0, self.COLS-1)
            pos = (r, c)
            if pos not in (self.start, self.goal) and self.grid[r][c] == 0:
                self.grid[r][c] = 1
                if pos in self.path[self.agent_idx:]:
                    cur = self.path[self.agent_idx-1] if self.agent_idx > 0 else self.start
                    new_path, vis, nv = self._algo_fn()(
                        self.grid, cur, self.goal,
                        self.ROWS, self.COLS, self._heuristic()
                    )
                    self.visited       |= vis
                    self.nodes_visited += nv
                    if new_path:
                        self.path       = new_path
                        self.agent_idx  = 0
                        self.status_msg = "Re-planned!"
                        self.status_col = C_YELLOW
                    else:
                        self.running_anim = False
                        self.status_msg   = "Path blocked!"
                        self.status_col   = C_RED

    def _advance_agent(self):
        if self.running_anim and self.path:
            if self.agent_idx < len(self.path):
                self.agent_idx += 1
            if self.agent_idx >= len(self.path):
                self.running_anim = False
                self.status_msg   = "Goal reached!"
                self.status_col   = C_GREEN

    def _draw_grid(self):
        path_set = set(self.path)
        for r in range(self.ROWS):
            for c in range(self.COLS):
                x, y = c*CELL, r*CELL
                rect = (x, y, CELL-1, CELL-1)
                pos  = (r, c)
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
                pygame.draw.rect(self.screen, col, rect, border_radius=2)
                pygame.draw.rect(self.screen, C_GRID, rect, 1, border_radius=2)

        if self.path and 0 < self.agent_idx <= len(self.path):
            ar, ac = self.path[self.agent_idx-1]
            pygame.draw.circle(self.screen, C_AGENT,
                               (ac*CELL + CELL//2, ar*CELL + CELL//2), CELL//2 - 3)

    def _draw_section_label(self, text, x, y):
        lbl = FONT_XS.render(text, True, C_LABEL)
        self.screen.blit(lbl, (x, y))

    def _draw_metric_box(self, x, y, w, label, value, val_color=None):
        box = pygame.Rect(x, y, w, 52)
        pygame.draw.rect(self.screen, C_PANEL2, box, border_radius=6)
        pygame.draw.rect(self.screen, C_GRID,   box, 1, border_radius=6)
        lbl = FONT_XS.render(label, True, C_LABEL)
        self.screen.blit(lbl, (x+8, y+7))
        col = val_color if val_color else C_TEXT
        val = FONT_XL.render(value, True, col)
        self.screen.blit(val, (x+8, y+22))

    def _draw_panel(self):
        px = self.COLS * CELL
        pygame.draw.rect(self.screen, C_PANEL, (px, 0, PANEL_W, self.WIN_H))
        pygame.draw.line(self.screen, C_GRID, (px, 0), (px, self.WIN_H), 2)

        mouse = pygame.mouse.get_pos()

        title = FONT_LG.render("PATHFINDING AGENT", True, C_ACCENT)
        self.screen.blit(title, (px+10, 10))

        sections = {
            self.btn_astar: "ALGORITHM",
            self.btn_manh:  "HEURISTIC",
            self.btn_run:   "CONTROLS",
            self.btn_rand:  "MAP",
            self.btn_wall:  "DRAW MODE",
            self.btn_dyn:   "DYNAMIC",
        }
        for b in self.all_buttons:
            if b in sections:
                self._draw_section_label(sections[b], px+10, b.rect.y - 13)
            b.draw(self.screen, mouse)

        my = self.metrics_y
        pygame.draw.line(self.screen, C_GRID, (px+8, my), (px+PANEL_W-8, my))
        my += 8

        self._draw_section_label("METRICS DASHBOARD", px+10, my)
        my += 16

        status_colors = {
            "Goal reached!": C_GREEN,
            "Path found!":   C_GREEN,
            "Re-planned!":   C_YELLOW,
            "No path found!":C_RED,
            "Path blocked!": C_RED,
        }
        st_col = status_colors.get(self.status_msg, C_TEXT)
        st_box = pygame.Rect(px+8, my, PANEL_W-16, 36)
        pygame.draw.rect(self.screen, C_PANEL2, st_box, border_radius=6)
        pygame.draw.rect(self.screen, st_col,   st_box, 1, border_radius=6)
        st_txt = FONT_MED.render(self.status_msg, True, st_col)
        self.screen.blit(st_txt, st_txt.get_rect(center=st_box.center))
        my += 44

        bw2 = (PANEL_W - 24) // 2

        self._draw_metric_box(px+8,       my, bw2, "NODES VISITED", str(self.nodes_visited), C_ACCENT)
        self._draw_metric_box(px+8+bw2+8, my, bw2, "PATH COST",     str(self.path_cost),     C_GREEN)
        my += 60

        exec_str = f"{self.exec_time:.1f} ms"
        self._draw_metric_box(px+8, my, PANEL_W-16, "EXECUTION TIME", exec_str, C_YELLOW)
        my += 60

        pygame.draw.line(self.screen, C_GRID, (px+8, my), (px+PANEL_W-8, my))
        my += 8

        self._draw_section_label("LEGEND", px+10, my)
        my += 14

        legend = [
            (C_START,   "Start node"),
            (C_GOAL,    "Goal node"),
            (C_WALL,    "Wall / obstacle"),
            (C_VISITED, "Visited nodes"),
            (C_PATH,    "Final path"),
            (C_AGENT,   "Agent"),
        ]
        for col, lbl in legend:
            pygame.draw.rect(self.screen, col, (px+10, my+1, 12, 12), border_radius=2)
            self.screen.blit(FONT_XS.render(lbl, True, C_TEXT), (px+26, my))
            my += 16

        my += 4
        pygame.draw.line(self.screen, C_GRID, (px+8, my), (px+PANEL_W-8, my))
        my += 8
        for h in ["ENTER=Run   C=Clear", "R=Random  DEL=Reset", "D=Dynamic Mode"]:
            self.screen.blit(FONT_XS.render(h, True, C_LABEL), (px+10, my))
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
        ANIM_DELAY = 80
        DYN_DELAY  = 300
        last_anim  = pygame.time.get_ticks()
        last_dyn   = pygame.time.get_ticks()
        mouse_held = False
        erase_mode = False

        while True:
            now = pygame.time.get_ticks()

            if self.running_anim and now - last_anim > ANIM_DELAY:
                self._advance_agent()
                last_anim = now

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
                    if mx < self.COLS * CELL:
                        mouse_held = True
                        erase_mode = (event.button == 3)
                        self._handle_click((mx, my), erase=erase_mode)
                    else:
                        if self.btn_astar.clicked(event.pos): self.algo = "astar"
                        elif self.btn_gbfs.clicked(event.pos): self.algo = "gbfs"
                        elif self.btn_manh.clicked(event.pos): self.heur = "manhattan"
                        elif self.btn_eucl.clicked(event.pos): self.heur = "euclidean"
                        elif self.btn_run.clicked(event.pos):
                            self.clear_path(); self.run_search()
                        elif self.btn_step.clicked(event.pos):
                            if not self.path:
                                self.run_search(); self.running_anim = False
                            else:
                                self._advance_agent()
                        elif self.btn_clear.clicked(event.pos): self.clear_path()
                        elif self.btn_rand.clicked(event.pos): self.random_map()
                        elif self.btn_reset.clicked(event.pos): self.reset_grid()
                        elif self.btn_wall.clicked(event.pos): self.placing = "wall"
                        elif self.btn_start.clicked(event.pos): self.placing = "start"
                        elif self.btn_goal.clicked(event.pos): self.placing = "goal"
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
            self.screen.fill(C_BG)
            self._draw_grid()
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    App().run()