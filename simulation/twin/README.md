# DELTA's physics twin

The drone built part by part in [MuJoCo](https://mujoco.org) — every part at
its weight and place, the motors from bench data, the gimbal and cage — and
a flight brain (PX4's controller, rebuilt small) that flies it. Separate from
the Habitat simulation: no Habitat, no conda, no big graphics card — only
Python and pip. MuJoCo's pip package supports Windows, Mac and Linux; the
twin was built and tested on Ubuntu under WSL.

## Set up (once)

Python 3.10 or newer. From the `simulation` folder:

```bash
python -m venv venv
```

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
pip install -r twin/requirements.txt
```

Mac or Linux:

```bash
source venv/bin/activate
pip install -r twin/requirements.txt
```

About 30 MB of packages.

## Fly it yourself

```bash
python -m twin.play
```

One window: the drone in a room, its numbers in the top left, help in the
bottom right (H hides it).

| Keys | | Mouse | |
|---|---|---|---|
| Up / Down | forward / back | left drag | turn the view around the drone |
| Left / Right | turn | right drag | slide the view |
| 1 / 0 (or Page Up / Down) | up / down — 1 on the floor takes off | scroll | zoom |
| Shift | faster | double-click | pick a thing (again: let go) |
| Space | stop and hold | Ctrl + right drag | push what is picked (the drone too) |
| L | land | Ctrl + left drag | twist it |
| G | a gust from the side | | |
| R | start again | F / C | view follows the drone or not / view back on it |
| Esc | quit | | |

A key asks for a speed; letting go stops the drone where it is (PX4's
Position mode). Rolling on the floor isn't in yet: on the floor with the
motors off it just sits.

## Fly a script, get numbers, a plot and a film

```bash
python -m twin.fly twin/flights/hover.json --out ~/delta-twin-out/hover --film
```

Writes `report.json` (power, flight time, overshoot, landing speed...),
`flight.png`, and with `--film` a `flight.mp4` (needs `ffmpeg` on the PATH;
without it you still get pictures of each step). Scripts are lists of steps
— `takeoff`, `hold`, `goto`, `yaw`, `push`, `land`; see the top of `fly.py`.

## Change the drone or the room

- The drone: `../config/body.json` — each part's weight, place and size, the
  motors' bench data, the gimbal, the cage. Every number says whether it is
  measured, a listing, or a guess.
- The room: `ROOM` at the top of `play.py`.

Close the window and start it again to see the change.

## Files

| | |
|---|---|
| `body.py` | `config/body.json` → the MuJoCo model |
| `motors.py` | the motors and props, from bench data: speed, thrust, power |
| `flight.py` | the flight brain: position → velocity → attitude → rates → motors |
| `fly.py` | fly a script, log it, plot it, film it |
| `play.py` | fly it yourself |
| `play_check.py` | Linux / WSL only: drives the play window with real key and mouse events and checks each one works |
