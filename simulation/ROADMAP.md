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

## Next — the replica, one phase at a time

Measured on tests/scripts/replica.json with {"score_replica": ...}: true depth
rendered from every place the lens looked from, against the brain's cubes.
With the true position the replica is 0.95; with RTAB-Map's 0.63. So:

### A. Scan where you stand
- **What:** a scan routine on one key (and a script step): turn 360° on the
  ground with the cameras tilted up, fly up, turn 360° with them tilted
  down, come back down. Your idea; it is what the inspection drones do at a
  point of interest.
- **Why:** rolling at 15 cm the cameras never see table tops, shelves or the
  upper half of a room; from up high they never see under things.
- **Test:** completeness of the replica before and after one scan at the
  same spot; seconds and metres flown per scan.

### B. Localization good enough for the replica
- **What:** the 0.95 → 0.63 gap: walls drawn twice because the position
  drifted 0.3–0.8 m between passes.
- **How:** first the map check that is already in the brain (compare what
  depth sees with the map, correct the position), measured on its own; then
  RTAB-Map's own scan matching against depth.
- **Test:** replica F-score and position error, with and without it.

### C. Explore and scan on its own
- **What:** go to the edge of the unexplored by the cheapest route, scan
  there (A), repeat until nothing is left (frontier exploration; the
  next-best-view work for 3D).
- **Test:** % of the house's surfaces in the replica per minute and per
  metre flown.

### D. Real depth
- Depth Anything instead of the true depth: every score above again.

### 3. When the house changes
- **What:** the map notices a thing added (a fridge, a person) and a thing
  taken away.
- **Why:** the glasses failed both ways: added after mapping → not marked;
  taken away → stayed in the map.
- **How:** comes from 1 — a ray that now passes through where the fridge
  was clears it; a new surface where rays used to pass marks it. Test
  scenario ported from the glasses (`run_objects.py`: keys add and remove a
  fridge, person, box).
- **Test:** seconds of looking until the change shows; cells still wrong
  after.
- **Connected to:** 1; the same scenario later tests 4.

### 4. Objects (YOLO), remembered properly
- **What:** "a sofa here, a fridge there" on the map.
- **Why:** rooms by use (kitchen, bedroom) and "go to the fridge" need it.
- **How:** each detection becomes a 3D box (mask + depth), matched to what is
  already remembered by class and overlap; trusted only after several views
  from different places; dropped when it should have been visible and was
  not (what the glasses lacked: clusters of guesses that never went away).
  Starts from the glasses' `objects.py`.
- **Test:** against HSSD's own object list for the house: found within 0.5 m
  (recall), false objects (precision), and the add/remove scenario of 3.
- **Connected to:** 1 (depth and visibility), 3; feeds 5 and 6.

### 5. Rooms by use
- **What:** "kitchen", "living room", "bedroom 2" — including splitting an
  open-plan space into kitchen / dining / living by what is in each part.
- **Why:** the robot-vacuum map, and "go to the kitchen".
- **How:** room kind from its objects (a stove and a fridge: kitchen; a bed:
  bedroom); open-plan zones around their objects; numbered when repeated.
- **Test:** against HSSD's room names — today 0.59–0.72 precision by use,
  because open plan cannot be split by walls.
- **Connected to:** rooms v1 ✓, 4; feeds 6.

### 6. Go to a room
- **What:** click or say "bedroom 2", "the kitchen", "the fridge"; the drone
  goes there on its own map.
- **How:** a route on the brain's map through the doors; rolling where it
  can. (The click-to-plan and autopilot exist; this adds names and rooms.)
- **Test:** arrived or not; route length vs the true shortest path.
- **Connected to:** 1, 5, 7.

### 7. Don't bump into things
- **What:** slow down near obstacles and stop 0.5 m away; then go around or
  fly over, by the drone's own size.
- **How:** speed limit √(2·a·(d − 0.5 m)) from depth in the corridor ahead;
  around / over / under chosen by cost (flying ≈ 4–5× rolling per metre).
- **Test:** contacts (must be 0), stop distance, success past obstacles.
- **Connected to:** 1; needed before 8.

## Later
- **Water** (pools, ponds): depth and the ultrasonic sensor can take water
  for a floor. Never land on a surface not recognised as solid.
- Energy model of the real hardware; the hardware itself.
