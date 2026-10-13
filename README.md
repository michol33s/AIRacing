# AI Car Racing

A simple Pygame racing simulation where NEAT evolves neural networks over generations.

## Files
- `main.py` — starts evolution and evaluates each generation.
- `game.py` — draws the oval track and the on-screen information.
- `car.py` — car movement, five sensors, seven inputs, and collision detection.
- `config-neat.txt` — NEAT configuration for 7 inputs and 2 outputs.
- `requirements.txt` — Python dependencies.

## Install and run
Install Python 3.10 or newer, open a terminal in this folder, then run:

```bash
python -m pip install -r requirements.txt
python main.py
```

## Neural network inputs
1. Far-left wall sensor (-70 degrees), normalized to 0..1
2. Left wall sensor (-35 degrees), normalized to 0..1
3. Front wall sensor (0 degrees), normalized to 0..1
4. Right wall sensor (+35 degrees), normalized to 0..1
5. Far-right wall sensor (+70 degrees), normalized to 0..1
6. Speed divided by maximum speed (0..1)
7. Track tangent direction error relative to car heading, normalized to -1..1

## Neural network outputs
1. Steering (-1..1)
2. Throttle (-1..1): positive accelerates, negative brakes

Press **S** to show/hide sensor rays; **Esc** quits the current generation and stops training.

The cars start with randomly initialized networks. They are expected to crash at first; NEAT evolves the networks over generations. This is a simple arcade simulation on an elliptical track, not a realistic driving simulator.
