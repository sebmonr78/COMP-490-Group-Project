# DELTA — what we add, one feature at a time

Each feature says **what** it is, **why** we need it, **how** it will work,
**how we test it** (against the simulator's ground truth — the truth only
ever examines, it never feeds the brain), and what it is **connected to**.
One at a time: finish it, measure it, try it by hand, then the next.

## Done

| Feature | Measured |
|---|---|
| Localization: RTAB-Map + IMU heading + cage axle encoder | turns, climbs, flying: a few cm per motion; blind against a wall: 0.76 m → stays put |
| Height: down-facing rangefinder + flight controller's climb rate | 1–2 cm; table and stairs recognised as "something under me" |
| Camera tilt (I/K/O), camera marker, height box | |
| Rooms v1: the 2D map split at doors (indoor/outdoor taken out again: not needed) | by walls: 0.98 / 0.98 (true pose), 0.88 / 0.85 (RTAB-Map) |
| Bigger house (102816036: 29 x 21 m, 36 rooms) | 7 rooms, 6 doors found on a first drive |
| The map shows what is there: solid layer 0.08–1.8 m, white; lilac = roll under | edges found 0.62 / right 0.77 (true pose); 0.35 / 0.65 (RTAB-Map) |
| **The 3D replica**: coloured 5 cm cubes from every view, rebuilt on loop closures; its own panel (drag, scroll, F); saved as a .ply; window 1920x1080, resizable | true pose: accuracy 2.2 cm, F-score 0.95 at 5 cm; RTAB-Map: 4.0 cm, F 0.63 |
| The ceiling: the brain finds it in its own cubes and knows its height; the replica view takes it off | median error 3.1 cm, 89-92% of patches within 10 cm (2.80 m found, 2.78 m true) |
| **Runs light**: the replica drawn on its own thread and only when it changes, the map 4 times a second, no unused depth camera, views 8x cheaper; `--replica later` + `B` builds it only when asked; window fills the screen (4K: 1.5 px a point) | window 2-9 -> 21-23 loops/s (1080p), 14-20 (4K, full screen); a drive that never finished in 20 min now takes 2.7 min |

## Next — in this order

### 1. Run light (continues)
- **Left:** RTAB-Map uses 2.5-3.6 CPU cores, more than everything else
  together. Fewer features, or finding them on a half-size image, would cut
  it -- but only if localization stays as good: measured with
  tools/repeat.sh (localization differs run to run), never on one run.
- Habitat drawing the pictures (the cameras and the view behind the drone)
  is most of what remains in the window; on the real drone that cost is
  not there at all.

### 2. A 2.5D map to move by
- **What:** for every 5 cm of floor: is it free to roll on, how high is
  what stands there (can I roll under it, fly over it), and how high is
  the ceiling over it. A few numbers a cell -- 3 MB for this whole house.
- **Why:** this is what moving needs, while flying: roll, roll under, fly
  over, or how high can I go. The 3D replica is for people to look at; it
  can wait for the end of the task.
- **How:** from what the brain already has: the rolling and solid layers,
  the rangefinder's surfaces, the ceiling.
- **Test:** against the house: heights of tables and beds, clearance under
  them, ceiling -- each within a few cm.

### 3. Explore when nothing is mapped yet
- **What:** the drone maps a house on its own. Ways to try, side by side:
  roll and turn 360 at stops; your scan -- 360 on the ground with the
  cameras up, fly up, 360 with them down, land, roll on; and going to the
  edge of the unexplored (frontier exploration) and scanning there.
- **Test:** for each: % of the house mapped, map and replica accuracy,
  minutes, and energy (metres flown count 4-5x metres rolled).
- **Connected to:** 2 (where it can go), the replica (how complete).

### 4. Real depth, and nothing true left in the brain
- **What:** depth from the cameras instead of the simulator's.
- **Which:** the cameras are a stereo pair a known distance apart, so depth
  from matching the two pictures is metric by itself, and the OAK board
  the cameras sit on computes it on its own chip -- it costs the laptop
  nothing. Depth Anything 3 guesses from one picture and needs scaling;
  it may fill in blank walls stereo cannot match. Both are scored against
  the truth, then the better one (or both together) is used.
- **Test:** depth error by distance; then every score above again, with
  the brain using nothing true at all.

### 5. Localization good enough for the replica
- **What:** the 0.95 -> 0.53-0.63 gap: walls drawn twice because the position
  drifted 0.1-0.8 m between passes.
- **How:** the map check that is already in the brain, measured on its
  own; then RTAB-Map's own scan matching against depth.

### 6. When the house changes
- **What:** the map notices a thing added (a fridge, a person) and a thing
  taken away.
- **Why:** the glasses failed both ways: added after mapping → not marked;
  taken away → stayed in the map.
- **How:** comes from the map — a ray that now passes through where the fridge
  was clears it; a new surface where rays used to pass marks it. Test
  scenario ported from the glasses (`run_objects.py`: keys add and remove a
  fridge, person, box).
- **Test:** seconds of looking until the change shows; cells still wrong
  after.
- **Connected to:** the map; the same scenario later tests 7.

### 7. Objects (YOLO), remembered properly
- **What:** "a sofa here, a fridge there" on the map.
- **Why:** rooms by use (kitchen, bedroom) and "go to the fridge" need it.
- **How:** each detection becomes a 3D box (mask + depth), matched to what is
  already remembered by class and overlap; trusted only after several views
  from different places; dropped when it should have been visible and was
  not (what the glasses lacked: clusters of guesses that never went away).
  Starts from the glasses' `objects.py`.
- **Test:** against HSSD's own object list for the house: found within 0.5 m
  (recall), false objects (precision), and the add/remove scenario of 6.
- **Connected to:** 4 (depth), 6; feeds 8 and 9.

### 8. Rooms by use
- **What:** "kitchen", "living room", "bedroom 2" — including splitting an
  open-plan space into kitchen / dining / living by what is in each part.
- **Why:** the robot-vacuum map, and "go to the kitchen".
- **How:** room kind from its objects (a stove and a fridge: kitchen; a bed:
  bedroom); open-plan zones around their objects; numbered when repeated.
- **Test:** against HSSD's room names — today 0.59–0.72 precision by use,
  because open plan cannot be split by walls.
- **Connected to:** rooms v1 ✓, 7; feeds 9.

### 9. Go to a room
- **What:** click or say "bedroom 2", "the kitchen", "the fridge"; the drone
  goes there on its own map.
- **How:** a route on the brain's map through the doors; rolling where it
  can. (The click-to-plan and autopilot exist; this adds names and rooms.)
- **Test:** arrived or not; route length vs the true shortest path.
- **Connected to:** 2, 8, 10.

### 10. Don't bump into things
- **What:** slow down near obstacles and stop 0.5 m away; then go around or
  fly over, by the drone's own size.
- **How:** speed limit √(2·a·(d − 0.5 m)) from depth in the corridor ahead;
  around / over / under chosen by cost (flying ≈ 4–5× rolling per metre).
- **Test:** contacts (must be 0), stop distance, success past obstacles.
- **Connected to:** 2 (heights), 4 (depth).

## Later
- **Water** (pools, ponds): depth and the ultrasonic sensor can take water
  for a floor. Never land on a surface not recognised as solid.
- Energy model of the real hardware; the hardware itself.
