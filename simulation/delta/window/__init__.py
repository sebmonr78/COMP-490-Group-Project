"""The simulated drone's window -- the half of DELTA that is not the brain.

    app.py        the simulated drone and its window: drives it (keys, roam,
                  script), sends the brain its sensors' readings, draws what
                  the brain sends back
    examiner.py   the only code that reads the simulator's truth: the grey
                  true map, the green ring, the "truth" column, the scores
    link.py       the cable to the brain
    script.py     test drives from tests/scripts/*.json
    view3d.py     drawing the 3D replica panel
    colours.py    the window's colours

Started by demo/run_drone.py (bash run_drone.sh starts the brain too).
"""
