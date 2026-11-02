# DELTA — the physical drone

DELTA is a drone inside a 45 cm ball-shaped cage that can roll on the floor
and fly. This folder is the drone as **hardware**, simulated: every part
from the parts list at its real weight and place, the motors from real bench
data, the gimbal and the cage, real physics (MuJoCo) — and a flight brain
that flies it like a real drone does (PX4's controller, rebuilt small).

It answers questions like: how long does it fly on this battery? does it
hover steadily? how hard does it lean to fly forward? what happens in a gust,
or when it bumps the table? And it lets you fly it yourself.

It is separate from the Habitat simulation in `simulation/` (the house, the
cameras, the mapping brain). No Habitat, no conda, no big graphics card here
— only Python and pip.

| | |
|---|---|
| Weight | 1.467 kg (cage, rings and battery included) |
| Hover | 21% of full thrust, 15,060 rpm, 405 W |
| Hover time | 14.3 min on the 4S2P Li-ion battery (121 Wh, 80% used) |
| Fly 2 m forward | leans 9°, stops within 1 cm |
| Runs | ~11× faster than real time without the window |

## 1. Set up (once)

You need **Python 3.10 or newer**. In a terminal, from this folder
(`physical-drone`):

```bash
python -m venv venv
```

**Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, allow scripts for your user once
with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activating
and use `venv\Scripts\python` wherever this page says `python`.

**Mac or Linux**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

About 30 MB of packages. Next time, only activate the `venv` (the first line)
before running.

## 2. Fly it yourself

```bash
python -m twin.play
```

A window opens: the drone on the floor of a room with a table, a sofa, a
shelf, a low step, a wall with a doorway, and boxes and a ball you can knock
around. Its numbers are in the top left; the help is in the bottom right
(H hides it). Press **1** to take off.

| Keys | | Mouse | |
|---|---|---|---|
| Up / Down | forward / back | left drag | turn the view around the drone |
| Left / Right | turn | right drag | slide the view |
| 1 / 0 (or Page Up / Down) | up / down — 1 on the floor takes off | scroll | zoom (wheel up = in) |
| Shift | faster | double-click | pick a thing (again: let go) |
| Space | stop and hold | Ctrl + right drag | push what is picked — the drone too |
| L | land (motors stop on the floor) | Ctrl + left drag | twist it |
| G | a gust from the side | | |
| R | start again | F | the view follows the drone, or stays |
| Esc | quit | C | the view back on the drone |

A key asks for a speed; letting go stops the drone where it is, like a real
drone in position mode. The numbers in the corner:

- **height, speed, lean, heading** — where it is and how it sits
- **power, top motor** — watts drawn now, and the busiest motor's share of
  full thrust
- **battery, time left** — energy used so far, and minutes left at this power
- **touching** — what the cage is touching (the table, a wall...)
- **keys** — the keys the window sees held down
- **sim speed** — 1.00 means the physics keeps up with the clock

## 3. Fly a script: numbers, a plot, a film

```bash
python -m twin.fly twin/flights/hover.json --out ~/delta-twin-out/hover --film
```

`hover.json` takes off, hovers 15 s, flies forward 2 m, climbs 1 m, turns 90°,
rides a gust, flies back and lands. In the `--out` folder you get:

- `report.json` — the numbers: power, hover time, overshoot, landing speed...
- `flight.png` — height, position, lean and heading, each motor, power,
  over time
- `flight.mp4` and a picture of each step (with `--film`; the film needs
  `ffmpeg` on the PATH, the pictures don't)

A script is a list of steps, done in order:

```json
[
 {"takeoff": 1.0},
 {"hold": 5},
 {"goto": [2.0, 0.0, 1.0], "name": "forward 2 m"},
 {"yaw": 90},
 {"push": [0.0, 3.0, 0.0], "for": 0.5},
 {"land": true}
]
```

Heights are of the ball's centre above the floor (0.225 m when it sits on
it); x is forward, y left, z up, in metres. `push` is newtons on the cage.
Copy `twin/flights/hover.json`, change it, and pass your file instead.

## 4. Change the drone or the room

- **The drone:** `config/body.json` — each part's weight, place and size, the
  motors' bench data, the gimbal, the cage, the air drag. Every number says
  whether it is measured, from a listing, or a guess. Change one, run again,
  and the physics follows: a heavier battery hovers on more power and for
  less time; a part moved forward makes the front motors work harder.
- **The room:** `ROOM` at the top of `twin/play.py` (MuJoCo's XML: boxes,
  spheres, capsules).

## 5. What is in it, and what isn't yet

In: the body part by part; the four motors (thrust from speed, power from
thrust, spin-up lag, the twist that turns the frame); the three-bearing
gimbal, so the cage turns freely around the drone; air drag; the cage
bumping into things; PX4's flight controller.

Not yet, in the order they are planned:

1. **Sensors** — the flight brain still reads the drone's true position and
   speed; next it gets a gyro, an accelerometer, a height laser and an
   optical flow sensor instead, as the real one will.
2. **Rolling** — on the floor with the motors off it just sits.
3. **The battery** — its voltage dropping as it empties, and current limits.
4. **Ground effect, walls, crashes.**
5. **The mapping sensors**, and connecting it to the mapping brain in
   `simulation/`.

The motor numbers come from the ECO III 2207 2400KV bench table (EMAX
publishes none for our ECO II); they should be replaced by our own thrust
stand test when the motors arrive.

## Files

| | |
|---|---|
| `config/body.json` | the drone, part by part |
| `twin/body.py` | `config/body.json` → the MuJoCo model |
| `twin/motors.py` | the motors and props, from bench data |
| `twin/flight.py` | the flight brain: position → speed → lean → turn rates → motors |
| `twin/fly.py` | fly a script, log it, plot it, film it |
| `twin/play.py` | fly it yourself |
| `twin/play_check.py` | Linux / WSL only: drives the play window with real key and mouse events and checks each one works (`python -m twin.play_check`) |
| `twin/flights/` | flight scripts |

## Trouble

- **`No module named twin`** — run from the `physical-drone` folder, with
  `python -m twin.play` (not `python twin/play.py`).
- **The window is slow under WSL** — the title says `[WARN:COPY MODE]`:
  restart WSL's display (`wsl --shutdown`, then start again). Inside WSL the
  window draws on the CPU on purpose: there it is faster than WSL's GPU
  bridge for MuJoCo.
- **Keys do nothing** — click the drone's window first so it has the keys;
  the `keys` line in the corner shows what it sees.
