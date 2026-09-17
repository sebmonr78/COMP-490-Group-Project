"""Reusable pieces of DELTA's simulation side.

DELTA -- Detecting Environmental Layout and Tracking Alterations.

Anything in here must work in any Habitat scene, not just the one
demo/open_environment.py happens to open. That rule is what will let this
code survive the move to Isaac Sim later: a module that only knows about
one house has to be rewritten, and a module that asks the simulator
questions does not.
"""
