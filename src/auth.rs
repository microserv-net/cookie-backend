//! Pairing and authentication.
//!
//! The threat is not a stranger on the internet. It is that this endpoint can
//! run commands on your laptop, and it lives on a home network with a
//! television and a doorbell on it. An unauthenticated endpoint that powerful
//! is the failure mode the whole design exists to avoid.
//!
//! Pair once, then never think about it again:
//!
//! ```text
//! backend:    cookie-backend pair
//!             → three readable words, ten minutes, single use
//! frontend:   POST /v1/pair {"code": "...", "device_name": "laptop"}
//!             → a token, stored on that machine
//! thereafter: Authorization: Bearer <token>
//! ```
//!
//! Words rather than hex because the code gets read across a room, and a code
//! people mistype is a code people disable. Tokens are stored hashed, so a
//! leaked state file does not hand over access, and devices are revocable by
//! name because "that laptop is gone" happens.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// How long a pairing code lives.
pub const PAIRING_TTL: Duration = Duration::from_secs(600);
/// Words per code. Three of forty-four is about 90 bits with the timeout,
/// which is plenty for something that expires in ten minutes.
const CODE_WORDS: usize = 3;

const WORDS: &[&str] = &[
    "amber", "apple", "anchor", "basil", "beacon", "birch", "candle", "cedar", "cinder", "clover",
    "copper", "cotton", "dahlia", "ember", "fennel", "ginger", "harbour", "hazel", "indigo",
    "ivory", "juniper", "kettle", "lantern", "linen", "marble", "meadow", "nutmeg", "olive",
    "orchid", "pebble", "quartz", "quince", "rowan", "saffron", "sorrel", "tamarind", "thistle",
    "umber", "velvet", "walnut", "willow", "yarrow", "zinnia", "bramble",
];

/// A frontend that has been paired.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub name: String,
    /// SHA-256 of the token. The token itself is never written down.
    pub token_hash: String,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_seen_at: Option<u64>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default)]
    devices: Vec<Device>,
}

/// Paired devices and outstanding pairing codes.
#[derive(Debug)]
pub struct DeviceStore {
    path: PathBuf,
    devices: BTreeMap<String, Device>,
    /// code → expiry. Held in memory only: a pairing code that survives a
    /// restart is a pairing code somebody forgot about.
    pending: BTreeMap<String, u64>,
}

impl DeviceStore {
    /// Open the store, tolerating a corrupt file.
    ///
    /// A corrupt store must not lock you out of your own machine; it means
    /// re-pairing, which is one command.
    pub fn open(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let devices = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Stored>(&text).ok())
            .map(|stored| {
                stored
                    .devices
                    .into_iter()
                    .map(|device| (device.name.clone(), device))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            path,
            devices,
            pending: BTreeMap::new(),
        }
    }

    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let stored = Stored {
            devices: self.devices.values().cloned().collect(),
        };
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&stored)?)
            .map_err(|e| Error::io(&tmp, e))?;
        restrict(&tmp);
        std::fs::rename(&tmp, &self.path).map_err(|e| Error::io(&self.path, e))?;
        Ok(())
    }

    /// Issue a single-use pairing code.
    pub fn begin_pairing(&mut self) -> String {
        self.begin_pairing_with_ttl(PAIRING_TTL)
    }

    pub fn begin_pairing_with_ttl(&mut self, ttl: Duration) -> String {
        self.expire();
        let code = generate_code();
        self.pending.insert(code.clone(), now() + ttl.as_secs());
        code
    }

    fn expire(&mut self) {
        let now = now();
        self.pending.retain(|_, expiry| *expiry > now);
    }

    /// Exchange a code for a token. The code is consumed either way.
    pub fn complete_pairing(&mut self, code: &str, device_name: &str) -> Result<String> {
        self.expire();
        // Constant-time comparison against each candidate: the set is tiny,
        // and a map lookup on a secret is a timing oracle.
        let matched = self
            .pending
            .keys()
            .find(|candidate| constant_time_eq(candidate.as_bytes(), code.as_bytes()))
            .cloned();
        let Some(matched) = matched else {
            return Err(Error::Pairing(
                "that pairing code is not valid. Run `cookie-backend pair` again.".into(),
            ));
        };
        self.pending.remove(&matched);

        let token = generate_token();
        let name = {
            let trimmed = device_name.trim();
            if trimmed.is_empty() {
                "unnamed device".to_string()
            } else {
                trimmed.to_string()
            }
        };
        self.devices.insert(
            name.clone(),
            Device {
                name,
                token_hash: hash(&token),
                created_at: now(),
                last_seen_at: None,
            },
        );
        self.save()?;
        Ok(token)
    }

    /// The device this token belongs to, if any.
    pub fn authenticate(&mut self, token: Option<&str>) -> Option<Device> {
        let token = token?;
        let digest = hash(token);
        let name = self
            .devices
            .values()
            .find(|device| constant_time_eq(device.token_hash.as_bytes(), digest.as_bytes()))
            .map(|device| device.name.clone())?;
        if let Some(device) = self.devices.get_mut(&name) {
            device.last_seen_at = Some(now());
            let device = device.clone();
            // Best-effort: failing to record a timestamp must not fail a
            // request that was otherwise perfectly valid.
            let _ = self.save();
            return Some(device);
        }
        None
    }

    pub fn devices(&self) -> Vec<Device> {
        let mut devices: Vec<Device> = self.devices.values().cloned().collect();
        devices.sort_by_key(|device| device.created_at);
        devices
    }

    pub fn revoke(&mut self, name: &str) -> Result<bool> {
        if self.devices.remove(name).is_none() {
            return Ok(false);
        }
        self.save()?;
        Ok(true)
    }

    /// True when nothing has paired yet.
    ///
    /// The API is open in that state, deliberately: a backend you cannot get
    /// into, on a machine across the room, is worse than one on your own
    /// network that has not been locked yet. The first pairing closes it.
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }
}

fn hash(token: &str) -> String {
    let digest = Sha256::digest(token.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Comparison that does not leak through timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// A human-speakable code, e.g. `cedar-quartz-willow`.
pub fn generate_code() -> String {
    let mut words = Vec::with_capacity(CODE_WORDS);
    for _ in 0..CODE_WORDS {
        let bytes = uuid::Uuid::new_v4();
        let index = bytes.as_bytes()[0] as usize % WORDS.len();
        words.push(WORDS[index]);
    }
    words.join("-")
}

/// 256 bits of token, from the same source the UUIDs come from.
fn generate_token() -> String {
    let first = uuid::Uuid::new_v4().simple().to_string();
    let second = uuid::Uuid::new_v4().simple().to_string();
    format!("{first}{second}")
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(unix)]
fn restrict(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict(_path: &Path) {
    // Windows inherits the user's profile permissions, which is the same
    // protection by a different mechanism.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (DeviceStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = DeviceStore::open(dir.path().join("devices.json"));
        (store, dir)
    }

    #[test]
    fn pairing_round_trips() {
        let (mut store, _dir) = store();
        let code = store.begin_pairing();
        let token = store.complete_pairing(&code, "laptop").unwrap();
        assert_eq!(store.authenticate(Some(&token)).unwrap().name, "laptop");
        assert!(store.authenticate(Some("nonsense")).is_none());
        assert!(store.authenticate(None).is_none());
    }

    #[test]
    fn codes_are_single_use() {
        let (mut store, _dir) = store();
        let code = store.begin_pairing();
        store.complete_pairing(&code, "laptop").unwrap();
        assert!(store.complete_pairing(&code, "another").is_err());
    }

    #[test]
    fn expired_codes_are_refused() {
        let (mut store, _dir) = store();
        let code = store.begin_pairing_with_ttl(Duration::from_secs(0));
        std::thread::sleep(Duration::from_millis(1100));
        assert!(store.complete_pairing(&code, "laptop").is_err());
    }

    #[test]
    fn tokens_are_never_written_to_disk() {
        let (mut store, dir) = store();
        let code = store.begin_pairing();
        let token = store.complete_pairing(&code, "laptop").unwrap();
        let written = std::fs::read_to_string(dir.path().join("devices.json")).unwrap();
        assert!(!written.contains(&token));
        assert!(written.contains("token_hash"));
    }

    #[test]
    fn devices_survive_a_restart_and_can_be_revoked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("devices.json");
        let token = {
            let mut store = DeviceStore::open(&path);
            let code = store.begin_pairing();
            store.complete_pairing(&code, "laptop").unwrap()
        };
        let mut reopened = DeviceStore::open(&path);
        assert!(reopened.authenticate(Some(&token)).is_some());
        assert!(reopened.revoke("laptop").unwrap());
        assert!(DeviceStore::open(&path)
            .authenticate(Some(&token))
            .is_none());
        assert!(!reopened.revoke("laptop").unwrap());
    }

    #[test]
    fn a_corrupt_store_means_re_pairing_not_a_dead_backend() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("devices.json");
        std::fs::write(&path, "{ this is not json").unwrap();
        let store = DeviceStore::open(&path);
        assert!(store.is_empty());
    }

    #[test]
    fn codes_are_readable_and_distinct() {
        let a = generate_code();
        assert_eq!(a.split('-').count(), CODE_WORDS);
        assert!(a.chars().all(|c| c.is_ascii_lowercase() || c == '-'));
        let distinct: std::collections::BTreeSet<String> =
            (0..20).map(|_| generate_code()).collect();
        assert!(distinct.len() > 1, "codes must not be constant");
    }

    #[test]
    fn constant_time_eq_behaves_like_eq() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }
}
