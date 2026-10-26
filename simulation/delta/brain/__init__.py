"""The glasses' computer: every box on the Project Method slide but the cameras.

One module per box, in the order a frame flows through them:

    rectify.py     Rectify and undistort            OpenCV
    depth.py       Estimate depth                   Depth Anything 3
    localize.py    Estimate wearer position         RTAB-Map
    detect.py      Detect and segment objects       YOLOE
    occupancy.py   Build occupancy grid             NumPy
    objects.py     Place objects in the map         labelled object memory
    navmap.py      Create safe 2D navigation map    OpenCV, NumPy
    planner.py     Plan a path                      NumPy
    guidance.py    Voice and audio guidance         templated cues
    speech.py        ... spoken with                PiperTTS
    goals.py       Interpret spoken goal            Gemma 4 E4B via Ollama
    hazard.py      The third loop (#5)              OpenCV SGBM

`loop.py` wires them together; `python -m delta.brain` runs it.

Nothing in here imports Habitat. The sim is one camera source among others:
when the OAK arrives, it replaces the sim and none of these files change.
"""
