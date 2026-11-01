//! Program status reports (`OSC 7501`): a program says whether it is idle, working,
//! finished, failed, or waiting on the user, and the terminal keeps the last report for
//! each record in a small table.
//!
//! ```text
//!   ESC ] 7501 ; state=blocked:kind=permission:app=terraform:msg=QXBwbHk/ ST
//!                └──────────── key=value pairs separated by ':' ────────────┘
//!
//!   body ──▶ parse ──────────────▶ Records ──────────────▶ summary ──▶ tab marker
//!            every pair checked;    one per id, the root    the most
//!            any failure drops      record has none         demanding state
//!            the whole report
//! ```
//!
//! A report replaces its record whole. Ids are `/`-separated paths, so `build/test` sits
//! under `build`, and `state=clear` removes a record together with everything under it.
//!
//! Only each record's id and state are kept. `msg`, `title` and `app` are checked,
//! because a bad one discards the report, and then dropped.

use crate::platform::bytes::base64_decode_into;

/// The most records one terminal keeps: the protocol's minimum.
const RECORD_LIMIT: usize = 64;

/// The longest sequence, `ESC ]` through the terminator, as sent. The grid checks it,
/// because only the parser knows the bytes it stripped.
pub(crate) const SEQUENCE_MAX: usize = 4096;

const KEY_MAX: usize = 16;
const APP_MAX: usize = 32;
const ID_MAX: usize = 128;
const ID_DEPTH: usize = 8;
const SEGMENT_MAX: usize = 32;
const MSG_ENCODED_MAX: usize = 2732;
const MSG_MAX: usize = 2048;
const TITLE_ENCODED_MAX: usize = 256;
const TITLE_MAX: usize = 192;

/// What a record says its program is doing, declared from least to most demanding,
/// which is the order [`Records::summary`] ranks by.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum State {
    Idle,
    Working,
    Done,
    Error,
    Blocked,
}

/// One record. The root record's id is empty.
#[derive(Clone, Copy)]
struct Record {
    id: [u8; ID_MAX],
    id_len: u8,
    state: State,
}

impl Record {
    const EMPTY: Record = Record {
        id: [0; ID_MAX],
        id_len: 0,
        state: State::Idle,
    };

    /// `None` only for an id past [`ID_MAX`], which [`parse_id`] has already refused.
    fn new(id: &[u8], state: State) -> Option<Record> {
        let mut bytes = [0; ID_MAX];
        bytes.get_mut(..id.len())?.copy_from_slice(id);
        Some(Record {
            id: bytes,
            id_len: u8::try_from(id.len()).ok()?,
            state,
        })
    }

    fn id(&self) -> &[u8] {
        self.id.get(..usize::from(self.id_len)).unwrap_or_default()
    }
}

/// One terminal's records, least recently updated first.
pub(crate) struct Records {
    slots: [Record; RECORD_LIMIT],
    len: usize,
}

impl Default for Records {
    fn default() -> Self {
        Records {
            slots: [Record::EMPTY; RECORD_LIMIT],
            len: 0,
        }
    }
}

/// What a report asks for, once every pair has passed.
enum Request<'a> {
    Set {
        id: &'a [u8],
        state: State,
    },
    /// Remove the record `id` names and everything under it, or every record for `None`.
    Clear(Option<&'a [u8]>),
}

impl Records {
    /// Apply one report body (everything after `7501;`), or nothing when it fails a check.
    pub(crate) fn apply(&mut self, body: &[u8]) {
        match parse(body) {
            Some(Request::Set { id, state }) => self.set(id, state),
            Some(Request::Clear(Some(id))) => self.retain(|r| !within(r.id(), id)),
            Some(Request::Clear(None)) => self.len = 0,
            None => {}
        }
    }

    fn records(&self) -> &[Record] {
        self.slots.get(..self.len).unwrap_or_default()
    }

    /// Move `id`'s record to the back, evicting the front one when the table is full.
    fn set(&mut self, id: &[u8], state: State) {
        let Some(record) = Record::new(id, state) else {
            return;
        };
        let from = match self.records().iter().position(|r| r.id() == id) {
            Some(index) => index,
            None if self.len == RECORD_LIMIT => 0,
            None => {
                self.len += 1;
                self.len - 1
            }
        };
        if let Some(tail) = self.slots.get_mut(from..self.len) {
            tail.rotate_left(1);
            if let Some(last) = tail.last_mut() {
                *last = record;
            }
        }
    }

    fn retain(&mut self, keep: impl Fn(&Record) -> bool) {
        let mut kept = 0;
        for index in 0..self.len {
            let Some(&record) = self.slots.get(index) else {
                break;
            };
            if keep(&record) {
                if let Some(slot) = self.slots.get_mut(kept) {
                    *slot = record;
                }
                kept += 1;
            }
        }
        self.len = kept;
    }

    /// The most demanding state across the records, ignoring `idle`.
    pub(crate) fn summary(&self) -> Option<State> {
        self.records()
            .iter()
            .map(|r| r.state)
            .filter(|&state| state != State::Idle)
            .max()
    }

    /// Drop the `done` and `error` records, once the user has seen the terminal.
    pub(crate) fn acknowledge(&mut self) {
        self.retain(|r| !matches!(r.state, State::Done | State::Error));
    }

    /// Drop the `working` and `blocked` records at a new shell prompt: whatever set them
    /// has left the foreground.
    pub(crate) fn prompt_started(&mut self) {
        self.retain(|r| !matches!(r.state, State::Working | State::Blocked));
    }
}

/// What the report in `body` asks for, or `None` when the protocol says to discard or
/// ignore it.
fn parse(body: &[u8]) -> Option<Request<'_>> {
    let mut state = None;
    let mut id = None;
    for pair in body.split(|&b| b == b':') {
        let mut halves = pair.splitn(2, |&b| b == b'=');
        let key = halves.next().unwrap_or_default().trim_ascii();
        // No `=`: not a pair, so no limit applies.
        let Some(value) = halves.next().map(<[u8]>::trim_ascii) else {
            if key == b"id" {
                return None;
            }
            continue;
        };
        // Limits come before characters, so an oversized key or value discards the
        // report even when it is malformed too.
        if key.len() > KEY_MAX || value.len() > value_max(key) {
            return None;
        }
        if key.is_empty() || !key.iter().all(u8::is_ascii_lowercase) {
            continue;
        }
        match key {
            // A malformed `id` discards the report rather than being skipped: without it
            // the report would land on the root record, and a `clear` on every record.
            b"id" => id = Some(parse_id(value)?),
            _ if !value.iter().all(|&b| is_value_byte(b)) => continue,
            b"state" => state = Some(value),
            b"msg" => check_text(value, MSG_MAX)?,
            b"title" => check_text(value, TITLE_MAX)?,
            // `kind`, `progress`, `app` and unknown keys are not stored.
            _ => {}
        }
    }
    let state = match state? {
        b"clear" => return Some(Request::Clear(id)),
        b"idle" => State::Idle,
        b"working" => State::Working,
        b"done" => State::Done,
        b"blocked" => State::Blocked,
        b"error" => State::Error,
        _ => return None,
    };
    Some(Request::Set {
        id: id.unwrap_or_default(),
        state,
    })
}

fn is_value_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"_.,+/=-".contains(&b)
}

/// `value` as an id path, or `None` when it breaks the grammar or a limit.
fn parse_id(value: &[u8]) -> Option<&[u8]> {
    let segment_ok = |(depth, segment): (usize, &[u8])| {
        depth < ID_DEPTH
            && (1..=SEGMENT_MAX).contains(&segment.len())
            && segment
                .iter()
                .all(|&b| b.is_ascii_alphanumeric() || b"_.+-".contains(&b))
    };
    (value.len() <= ID_MAX && value.split(|&b| b == b'/').enumerate().all(segment_ok))
        .then_some(value)
}

/// The most bytes a value of `key` may hold. [`parse_id`] checks an `id`'s limits.
fn value_max(key: &[u8]) -> usize {
    match key {
        b"msg" => MSG_ENCODED_MAX,
        b"title" => TITLE_ENCODED_MAX,
        b"app" => APP_MAX,
        _ => usize::MAX,
    }
}

/// Whether a base64 `msg` or `title` decodes to at most `decoded_max` bytes of UTF-8 with
/// no control characters.
fn check_text(value: &[u8], decoded_max: usize) -> Option<()> {
    let mut buffer = [0; MSG_MAX];
    let out = buffer.get_mut(..decoded_max)?;
    let len = base64_decode_into(value, out)?;
    let text = std::str::from_utf8(out.get(..len)?).ok()?;
    (!text.chars().any(char::is_control)).then_some(())
}

/// Whether `record` is `id` or sits under it, by whole segment: `builder` is not under
/// `build`.
fn within(record: &[u8], id: &[u8]) -> bool {
    record
        .strip_prefix(id)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(b"/"))
}

#[cfg(test)]
mod tests {
    use crate::program_status::*;

    fn apply_all(reports: &[impl AsRef<str>]) -> Records {
        let mut records = Records::default();
        for report in reports {
            records.apply(report.as_ref().as_bytes());
        }
        records
    }

    /// The table as `(id, state)` pairs, least recently updated first.
    fn table(records: &Records) -> Vec<(&str, State)> {
        records
            .records()
            .iter()
            .map(|r| (std::str::from_utf8(r.id()).unwrap(), r.state))
            .collect()
    }

    #[test]
    fn a_report_replaces_its_record_and_each_id_has_its_own() {
        let records = apply_all(&[
            "state=working:app=deploy",
            "state=working:id=us-east:progress=40",
            "state=blocked:kind=permission:id=eu-west",
            "state=done:id=us-east",
        ]);
        assert_eq!(
            table(&records),
            [
                ("", State::Working),
                ("eu-west", State::Blocked),
                ("us-east", State::Done),
            ]
        );
    }

    #[test]
    fn a_clear_removes_a_subtree_by_whole_segment_and_a_bare_one_removes_all() {
        let mut records = apply_all(&[
            "state=working",
            "state=working:id=build",
            "state=blocked:id=build/test",
            "state=done:id=builder",
        ]);
        records.apply(b"state=clear:id=build");
        assert_eq!(
            table(&records),
            [("", State::Working), ("builder", State::Done)],
            "build and build/test go, builder and the root stay"
        );
        records.apply(b"state=clear");
        assert!(table(&records).is_empty());
    }

    #[test]
    fn a_malformed_id_discards_the_report_rather_than_reaching_the_root() {
        let mut records = apply_all(&["state=blocked", "state=done:id=job"]);
        let before = format!("{:?}", table(&records));
        // Each `id` holds a byte outside the value set. Skipping just the pair would send
        // the first two to the root record and make the clear remove every record.
        for report in [
            "state=idle:id=eu west",
            "state=idle:id=café",
            "state=clear:id=job#1",
            "state=clear:id",
        ] {
            records.apply(report.as_bytes());
            assert_eq!(format!("{:?}", table(&records)), before, "{report}");
        }
    }

    #[test]
    fn an_id_outside_the_grammar_or_its_limits_is_ignored() {
        let segment = "s".repeat(32);
        let deepest = ["a"; 8].join("/");
        let mut records = apply_all(&[
            format!("state=working:id={segment}"),
            format!("state=working:id={deepest}"),
        ]);
        assert_eq!(
            records.records().len(),
            2,
            "the limits themselves are allowed"
        );

        let wide = format!("{segment}s");
        let deep = format!("{deepest}/a");
        let long = [segment.as_str(); 4].join("/") + "/x";
        for id in [
            "",            // a segment needs a byte
            "a,b",         // in the value set, not the id set
            "a//b",        // an empty segment
            wide.as_str(), // a 33-byte segment
            deep.as_str(), // nine levels
            long.as_str(), // 133 bytes
        ] {
            records.apply(format!("state=clear:id={id}").as_bytes());
            records.apply(format!("state=done:id={id}").as_bytes());
            assert_eq!(records.records().len(), 2, "{id:?}");
        }
    }

    #[test]
    fn free_text_must_decode_to_one_line_of_utf8_within_its_limits() {
        let applied = |report: &str| apply_all(&[report]).summary() == Some(State::Done);
        assert!(applied(
            "state=done:msg=SW5zdGFsbGluZyB1cGRhdGVz:title=VVMgRWFzdA"
        ));
        // A byte outside the value set makes a malformed pair, which is only skipped.
        assert!(applied("state=done:msg=SW5z!"));

        for (report, why) in [
            ("state=done:msg=A", "base64 that cannot decode"),
            ("state=done:msg=SW5z_", "a value-set byte outside base64"),
            ("state=done:msg=YQpi", "a newline"),
            ("state=done:msg=YcKF", "a C1 control, U+0085"),
            ("state=done:title=/w", "a byte that is not UTF-8"),
        ] {
            assert!(!applied(report), "{why}");
        }

        // 2048 decoded bytes in 2731 is the most a `msg` holds. "QUFB" is "AAA".
        let most = "QUFB".repeat(682) + "QUE";
        assert!(applied(&format!("state=done:msg={most}")));
        let decoded_over = "QUFB".repeat(683); // 2732 encoded, 2049 decoded
        assert!(!applied(&format!("state=done:msg={decoded_over}")));
        let encoded_over = "QUFB".repeat(683) + "QQ";
        assert!(!applied(&format!("state=done:msg={encoded_over}")));

        assert!(applied(&format!("state=done:title={}", "QUFB".repeat(64))));
        let title_over = "QUFB".repeat(64) + "QQ";
        assert!(!applied(&format!("state=done:title={title_over}")));
    }

    #[test]
    fn a_report_needs_a_known_state_and_ignores_unknown_keys() {
        for report in ["", "id=x", "state=", "state=paused", "state=Working"] {
            assert!(apply_all(&[report]).records().is_empty(), "{report:?}");
        }
        let records = apply_all(&["state=error:future=thing:kind=telepathy:progress=150"]);
        assert_eq!(table(&records), [("", State::Error)]);
    }

    #[test]
    fn pairs_are_trimmed_skipped_when_malformed_and_the_last_value_wins() {
        let records = apply_all(&[" state = working : id = a "]);
        assert_eq!(table(&records), [("a", State::Working)]);

        // No `=`, an empty key, a key that is not lowercase, a space inside a value.
        let records = apply_all(&["note:=x:State=idle:app=two words:state=done"]);
        assert_eq!(table(&records), [("", State::Done)]);

        let records = apply_all(&["state=working:state=blocked:id=a:id=b"]);
        assert_eq!(table(&records), [("b", State::Blocked)]);

        // An `app` outside its own set is only absent, but one past its length drops the
        // report, as does any key past sixteen bytes.
        assert_eq!(apply_all(&["state=done:app=a,b"]).records().len(), 1);
        let long_app = "a".repeat(33);
        assert!(apply_all(&[format!("state=done:app={long_app}")])
            .records()
            .is_empty());
        let long_key = "k".repeat(17);
        assert!(apply_all(&[format!("state=done:{long_key}=1")])
            .records()
            .is_empty());
        // The key limit binds keys, not a malformed token that never had one.
        let token = "t".repeat(17);
        assert_eq!(
            apply_all(&[format!("state=done:{token}")]).records().len(),
            1
        );
        let key_at_limit = "k".repeat(16);
        assert_eq!(
            apply_all(&[format!("state=done:{key_at_limit}=1")])
                .records()
                .len(),
            1
        );
    }

    #[test]
    fn past_the_limit_the_least_recently_updated_record_makes_room() {
        let mut records = Records::default();
        for n in 0..RECORD_LIMIT {
            records.apply(format!("state=working:id=r{n}").as_bytes());
        }
        // r0 is the oldest by creation, but updating it makes r1 the least recent.
        records.apply(b"state=blocked:id=r0");
        records.apply(b"state=done:id=new");
        let kept = table(&records);
        assert_eq!(kept.len(), RECORD_LIMIT);
        assert!(kept.iter().all(|&(id, _)| id != "r1"), "r1 made room");
        assert!(kept.contains(&("r0", State::Blocked)));
        assert_eq!(kept.last(), Some(&("new", State::Done)));
    }

    #[test]
    fn the_summary_is_the_most_demanding_state_and_idle_shows_nothing() {
        assert_eq!(apply_all(&["state=idle"]).summary(), None);
        let ladder = [
            ("state=working:id=a", State::Working),
            ("state=done:id=b", State::Done),
            ("state=error:id=c", State::Error),
            ("state=blocked:id=d", State::Blocked),
        ];
        let mut records = apply_all(&["state=idle"]);
        for (report, expected) in ladder {
            records.apply(report.as_bytes());
            assert_eq!(records.summary(), Some(expected), "{report}");
        }
        // The order reports arrive in does not matter.
        let mut reversed = Records::default();
        for (report, _) in ladder.iter().rev() {
            reversed.apply(report.as_bytes());
        }
        assert_eq!(reversed.summary(), Some(State::Blocked));
    }

    #[test]
    fn acknowledging_and_a_new_prompt_each_drop_their_own_states() {
        let reports = [
            "state=idle",
            "state=working:id=w",
            "state=done:id=d",
            "state=error:id=e",
            "state=blocked:id=b",
        ];
        let mut records = apply_all(&reports);
        records.acknowledge();
        assert_eq!(
            table(&records),
            [
                ("", State::Idle),
                ("w", State::Working),
                ("b", State::Blocked)
            ]
        );

        let mut records = apply_all(&reports);
        records.prompt_started();
        assert_eq!(
            table(&records),
            [("", State::Idle), ("d", State::Done), ("e", State::Error)]
        );
    }

    #[test]
    fn a_value_over_its_limit_discards_the_report_even_when_malformed() {
        // Each breaks a limit and holds a byte outside the value set. Skipping it as a
        // malformed pair would let the clear through.
        let msg = "QUFB".repeat(683) + "!"; // 2733 bytes
        let title = "QUFB".repeat(64) + "!"; // 257
        let app = "a".repeat(32) + "!"; // 33
        let key = "k".repeat(17);
        for pair in [
            format!("msg={msg}"),
            format!("title={title}"),
            format!("app={app}"),
            format!("{key}=!"),
            format!("{}=1", key.to_uppercase()),
            format!("{}!=1", "k".repeat(16)),
        ] {
            let mut records = apply_all(&["state=blocked"]);
            records.apply(format!("state=clear:{pair}").as_bytes());
            assert_eq!(table(&records), [("", State::Blocked)], "{pair:.12}");
        }

        // Within its limit, a malformed value is only skipped.
        let mut records = apply_all(&["state=blocked"]);
        records.apply(format!("state=clear:app={}!", "a".repeat(31)).as_bytes());
        assert!(records.records().is_empty());
    }
}
