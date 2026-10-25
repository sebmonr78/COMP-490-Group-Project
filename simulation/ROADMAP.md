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
| Rooms v1: the 2D map split at doors, indoor/outdoor per room | by walls: 0.98 / 0.98 (true pose), 0.88 / 0.85 (RTAB-Map) |

## Next

### 1. 3D map — the replica of the place
- **What:** a 3D map of 5 cm cubes (voxels), each coloured by what the camera
  saw, built from depth at the brain's own position.
- **Why:** there is no 3D map yet, only a 2D one at rolling height. Flying
  needs to know what is at every height; the 2D map wrongly fills in floor it
  never saw from the air (2.2% unsafe vs 0% rolling); and you want to see the
  place.
- **How:** every depth pixel is a ray: the cube it ends in gets "occupied",
  every cube it passed through gets "free". Evidence adds up (log-odds) and
  is capped, so a change can flip a cube in a few looks. Cubes hang off
  RTAB-Map's nodes, so a loop closure moves them (as the 2D map does now).
  The 2D rolling map, the floor plan and the rooms are then *cut from* the
  3D map — one source of truth instead of three.
- **Test** (true position and depth first, then real): share of cubes within
  5 cm of a real surface (precision), share of seen surfaces that got a cube
  (completeness); the 2D maps cut from it vs today's (fake obstacles,
  unsafe); ground vs air unsafe.
- **Connected to:** localization ✓ and height ✓ feed it; it feeds rooms,
  obstacle stops, navigation and exploration.

### 2. Look around the replica
- **What:** open the brain's 3D map and orbit it, next to the real house.
- **Why:** to see, not just score, what the drone built.
- **How:** the brain saves the coloured cubes; a viewer opens them (a 3D
  window, or a web page you can share).
- **Test:** side by side with the same view of the real house.
- **Connected to:** 1.

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

### 8. Explore on its own
- **What:** map the whole house without being driven, with as little
  flying and time as possible.
- **How:** go to the nearest edge of the unexplored (a frontier) by the
  cheapest route; at each stop, look around: a turn on the ground with the
  cameras tilted up, then fly up and turn with them tilted down, then land
  and roll on. Stop when there are no frontiers left.
- **Test:** % of the house mapped, minutes, metres flown vs rolled.
- **Connected to:** 1, 7.

### 9. Real depth
- **What:** Depth Anything instead of the simulator's true depth.
- **Test:** every test above again; what breaks gets fixed here.

### 10. Localization, better
- The map check (compare what depth sees with the map; already in the brain,
  switched off) — measured on its own.
- RTAB-Map slows as its map grows (perception ~6 → ~3 frames a second over
  one long run): tune its memory.

## Later
- **Water** (pools, ponds): depth and the ultrasonic sensor can take water
  for a floor. Never land on a surface not recognised as solid.
- Energy model of the real hardware; the hardware itself.
