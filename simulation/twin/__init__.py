"""DELTA's physics twin: the drone built part by part in MuJoCo, and the
brain that flies it.

Separate from the Habitat simulation (delta/): its own environment (conda
env `twin`), its own frame (+x forward, +y left, +z up). It is connected to
the mapping brain later, one piece at a time.

    body.py    config/body.json -> a MuJoCo model: parts, gimbal, cage
    motors.py  the motors and props, from bench data: speed, thrust, power
    flight.py  the flight brain: PX4's controller, rebuilt small
    fly.py     fly a script (twin/flights/*.json), log it, plot it, film it
"""
