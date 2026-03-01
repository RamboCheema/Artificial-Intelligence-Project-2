# Artificial-Intelligence-Project-2
# PATH_FINDING AGENT



A grid-based visual pathfinding simulation built with Python and Pygame.
Implements **A\* Search** and **Greedy Best-First Search (GBFS)** with real-time
visualization, dynamic obstacle spawning, and a live metrics dashboard.


## Requirements

-> Python 3.8 or higher
-> Pygame


## Installation

**Step 1 — Make sure Python is installed**

Open your terminal or command prompt and run:

```bash
python --version
```

If Python is not installed, download it from https://www.python.org/downloads/



**Step 2 — Install Pygame**

```bash
pip install pygame


If you have multiple Python versions installed, use:

bash
python -m pip install pygame




**Step 3 — Run the program**

Navigate to the folder where the file is saved and run:

bash
python pathfinding_agent.py
```



## How to Use

### Algorithm Selection
| Button | Description |
|---|---|
| A* Search | Uses f(n) = g(n) + h(n) — guarantees shortest path |
| Greedy BFS | Uses f(n) = h(n) — faster but not always optimal |

### Heuristic Selection
| Button | Description |
|---|---|
| Manhattan | Best for 4-directional grid movement |
| Euclidean | Straight-line distance estimate |

### Controls
| Button / Key | Action |
|---|---|
| ▶ Run Search / `ENTER` | Run the selected algorithm |
| ⏭ Step | Move agent one step at a time |
| ✕ Clear Path / `C` | Clear path and visited nodes, keep walls |
| ⊞ Random Map / `R` | Generate a random maze with ~28% wall density |
| ↺ Reset Grid / `DELETE` | Clear everything including walls |
| ⚡ Dynamic Mode / `D` | Toggle random obstacle spawning during agent movement |

### Drawing on the Grid
| Mode | How to activate | Action |
|---|---|---|
| Place Wall | Click **✏ Place Wall** | Left-click or drag to draw walls |
| Erase Wall | While in wall mode | Right-click to erase walls |
| Move Start | Click **S Move Start** | Click any empty cell to move start |
| Move Goal | Click **G Move Goal** | Click any empty cell to move goal |

---

## Color Legend

| Color | Meaning |
|---|---|
| Green (bright) | Start node |
| Red | Goal node |
| Dark grey | Wall / obstacle |
| Yellow | Frontier nodes (currently in priority queue) |
| Blue | Visited / expanded nodes |
| Green (light) | Final path |
| White circle | Agent |

---

## Features

- **A\* Search** — Finds the guaranteed shortest path using g(n) + h(n)
- **Greedy BFS** — Faster search using heuristic only, may not find shortest path
- **Manhattan and Euclidean heuristics** — Selectable before each run
- **Frontier visualization** — Yellow cells show nodes currently in the priority queue
- **Visited visualization** — Blue cells show all expanded nodes
- **Dynamic Mode** — Obstacles randomly spawn while agent is moving; agent automatically re-plans if its path is blocked
- **Interactive map editor** — Draw and erase walls freely by clicking or dragging
- **Random map generation** — Instantly generates a maze at 28% wall density
- **Real-time metrics dashboard** — Shows nodes visited, path cost, and execution time after every search



## Project Structure

```
pathfinding_agent.py    Main application — all code in a single file
README.md               This file




## Troubleshooting

**ModuleNotFoundError: No module named 'pygame'**

Run `pip install pygame` and try again. If it still fails, use:

```bash
python -m pip install pygame


**The window does not open**

Make sure you are running the file with Python 3, not Python 2:

bash
python3 pathfinding_agent.py


**Pygame installs but the program crashes immediately**

Try upgrading pygame:

```bash
pip install --upgrade pygame
```

---

## Institution

National University of Computer and Emerging Sciences (NUCES)
Chiniot-Faisalabad Campus
Artificial Intelligence Lab Assignment
