-- DELTA database schema (SQLite)
--
-- Seven tables, matching the schema section of the Phase 2 specification:
--
--   User -< Drone -< Session -< File
--                       |  \-< MapRecord >- File   (a map is stored in one file)
--                       |  \-< ObjectRecord -< ChangeRecord
--                       \-----< ChangeRecord (twice: previous and current session)
--
-- Conventions
--   * Timestamps are text in 'YYYY-MM-DD HH:MM:SS', UTC. db.now() produces
--     this format, and CURRENT_TIMESTAMP defaults match it.
--   * SQLite does not enforce VARCHAR lengths, so CHECK constraints guard the
--     fields that only allow certain values.
--   * Foreign keys are only enforced when each connection runs
--     PRAGMA foreign_keys = ON. db.connect() does that for you.
--   * Plain CREATE TABLE / CREATE INDEX only, so this file ports to
--     PostgreSQL with small edits (note: "User" is a reserved word there and
--     would need quoting or renaming).


CREATE TABLE User (
    user_id        INTEGER      PRIMARY KEY,
    username       VARCHAR(50)  NOT NULL UNIQUE,
    email          VARCHAR(100) NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE Drone (
    drone_id       INTEGER      PRIMARY KEY,
    serial_number  VARCHAR(50)  NOT NULL UNIQUE,         -- hardware ID
    owner_id       INTEGER      REFERENCES User(user_id), -- NULL while unpaired
    status         VARCHAR(20)  NOT NULL DEFAULT 'unpaired'
                   CHECK (status IN ('active', 'unpaired', 'maintenance'))
);


CREATE TABLE Session (
    session_id     INTEGER      PRIMARY KEY,
    drone_id       INTEGER      NOT NULL REFERENCES Drone(drone_id),
    started_at     TIMESTAMP    NOT NULL,
    ended_at       TIMESTAMP,                            -- NULL while in progress
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);


CREATE TABLE File (
    file_id        INTEGER      PRIMARY KEY,
    session_id     INTEGER      NOT NULL REFERENCES Session(session_id),
    drone_id       INTEGER      NOT NULL REFERENCES Drone(drone_id),
    owner_id       INTEGER      REFERENCES User(user_id), -- copy of Drone.owner_id, for fast queries
    file_type      VARCHAR(20)  NOT NULL
                   CHECK (file_type IN ('rgb_frame', 'depth_frame', 'map_export')),
    filename       VARCHAR(255) NOT NULL,
    storage_url    VARCHAR(500) NOT NULL,
    mime_type      VARCHAR(100),                         -- format, e.g. image/png
    size_bytes     BIGINT       CHECK (size_bytes IS NULL OR size_bytes >= 0),
    captured_at    TIMESTAMP,
    uploaded_at    TIMESTAMP
);


CREATE TABLE MapRecord (
    map_id         INTEGER      PRIMARY KEY,
    session_id     INTEGER      NOT NULL REFERENCES Session(session_id),
    file_id        INTEGER      NOT NULL UNIQUE REFERENCES File(file_id), -- the saved map data
    resolution     REAL         CHECK (resolution IS NULL OR resolution > 0), -- meters per grid cell
    origin_x       REAL,
    origin_y       REAL,
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE ObjectRecord (
    object_id      INTEGER      PRIMARY KEY,
    session_id     INTEGER      NOT NULL REFERENCES Session(session_id), -- first observed in
    label          VARCHAR(50)  NOT NULL,                -- e.g. 'chair'
    x              REAL         NOT NULL,                -- last known position, map coordinates
    y              REAL         NOT NULL,
    z              REAL         NOT NULL,
    confidence     REAL         CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    first_seen_at  TIMESTAMP    NOT NULL,
    last_seen_at   TIMESTAMP    NOT NULL,
    CHECK (last_seen_at >= first_seen_at)
);


CREATE TABLE ChangeRecord (
    change_id            INTEGER     PRIMARY KEY,
    object_id            INTEGER     REFERENCES ObjectRecord(object_id), -- NULL for area-level changes
    previous_session_id  INTEGER     NOT NULL REFERENCES Session(session_id), -- earlier scan
    current_session_id   INTEGER     NOT NULL REFERENCES Session(session_id), -- later scan
    change_type          VARCHAR(20) NOT NULL
                         CHECK (change_type IN ('moved', 'appeared', 'disappeared')),
    x                    REAL,                           -- location of the change
    y                    REAL,
    z                    REAL,
    description          VARCHAR(255),                   -- used for area-level changes
    detected_at          TIMESTAMP   NOT NULL,
    CHECK (previous_session_id <> current_session_id)
);


-- Indexes on the columns used for joins and lookups.
CREATE INDEX idx_drone_owner         ON Drone(owner_id);
CREATE INDEX idx_session_drone       ON Session(drone_id);
CREATE INDEX idx_file_session        ON File(session_id);
CREATE INDEX idx_map_session         ON MapRecord(session_id);
CREATE INDEX idx_object_session      ON ObjectRecord(session_id);
CREATE INDEX idx_object_label        ON ObjectRecord(label);
CREATE INDEX idx_change_object       ON ChangeRecord(object_id);
CREATE INDEX idx_change_previous     ON ChangeRecord(previous_session_id);
CREATE INDEX idx_change_current      ON ChangeRecord(current_session_id);
