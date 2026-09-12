-- CRAB-136: the lessons index schema, as a sqlx migration.
--
-- Compatibility rule: every object uses IF NOT EXISTS so running this
-- migration against a pre-CRAB-136 database (created by the old rusqlite
-- code with plain CREATE TABLE IF NOT EXISTS) is a clean no-op — the index
-- opens and searches without a rebuild. The JSONL lesson log remains the
-- source of truth; dropping `index.sqlite` loses nothing.

CREATE TABLE IF NOT EXISTS lessons (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- FTS5 needs its own rowids aligned with `lessons.rowid`; inserts manage
-- that explicitly (insert into the content table first, RETURNING rowid,
-- then into the index with the same rowid).
CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(id, text, kind, tags);
