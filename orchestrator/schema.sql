-- schema.sql - Algorithm 2.17 STATE-DB (SQLite). Times are epoch seconds (UTC).
CREATE TABLE IF NOT EXISTS sessions (
  session_id    TEXT PRIMARY KEY,
  src_ip        TEXT,
  username      TEXT,
  started       REAL,
  ended         REAL,
  n_lines       INTEGER,
  verdict       TEXT CHECK (verdict IN ('Human', 'Bot', 'Uncertain') OR verdict IS NULL),
  p_human       REAL,
  model_name    TEXT,
  features_json TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts          REAL NOT NULL,
  event_type  TEXT NOT NULL,
  src_ip      TEXT,
  session_id  TEXT REFERENCES sessions(session_id),
  action      TEXT NOT NULL,
  reason_json TEXT,
  latency_ms  REAL
);
CREATE TABLE IF NOT EXISTS decoys (
  decoy_id     TEXT PRIMARY KEY,
  svc          TEXT NOT NULL,
  profile_json TEXT,
  container_id TEXT,
  addr         TEXT,
  port         INTEGER,
  attacker_ip  TEXT,
  started      REAL,
  stopped      REAL,
  status       TEXT
);
CREATE TABLE IF NOT EXISTS rules (
  rule_id  INTEGER PRIMARY KEY AUTOINCREMENT,
  decoy_id TEXT REFERENCES decoys(decoy_id),
  src_ip   TEXT NOT NULL,
  dport    INTEGER NOT NULL,
  target   TEXT NOT NULL,
  comment  TEXT NOT NULL,
  added    REAL NOT NULL,
  removed  REAL
);
CREATE TABLE IF NOT EXISTS scans (
  scan_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            REAL NOT NULL,
  services_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_ip ON decisions(src_ip);
CREATE INDEX IF NOT EXISTS idx_rules_open   ON rules(removed);
