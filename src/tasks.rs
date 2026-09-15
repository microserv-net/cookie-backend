//! Tasks, and the part that makes one-model-at-a-time hardware bearable.
//!
//! A task is one thing the user asked for. It runs until it finishes, fails,
//! is cancelled, or is asked to **stand aside** — which is the interesting
//! case.
//!
//! Standing aside is cooperative. Nothing is preempted by force: a task calls
//! [`TaskManager::checkpoint`] at the points where it is already between
//! things, and waits there if something more urgent has arrived. The
//! checkpoints are exactly the boundaries where stopping costs nothing:
//!
//! * a model call has just returned,
//! * a model is about to be loaded or swapped,
//! * a tool is running and we are waiting on it,
//! * we are between steps of a plan.
//!
//! Never mid-generation, and never by discarding work. The distinction from
//! cancellation is the whole point: being asked to wait must not lose what
//! you have done, and being cancelled must only happen because someone asked.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::json;

/// How much of the machine a task expects to need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Weight {
    Light,
    Normal,
    Heavy,
}

impl Weight {
    pub fn as_str(self) -> &'static str {
        match self {
            Weight::Light => "light",
            Weight::Normal => "normal",
            Weight::Heavy => "heavy",
        }
    }
}

/// Where a task has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskState {
    Queued,
    Running,
    /// Yielded at a checkpoint so something more urgent could run. This state
    /// is what makes a one-model machine usable, and what the frontend shows
    /// as "paused" rather than "stuck".
    Suspended,
    Completed,
    Failed,
    Cancelled,
}

impl TaskState {
    pub fn is_active(self) -> bool {
        matches!(
            self,
            TaskState::Queued | TaskState::Running | TaskState::Suspended
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TaskState::Queued => "queued",
            TaskState::Running => "running",
            TaskState::Suspended => "suspended",
            TaskState::Completed => "completed",
            TaskState::Failed => "failed",
            TaskState::Cancelled => "cancelled",
        }
    }
}

/// A snapshot of one unit of work, as reported to the frontend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskInfo {
    pub id: String,
    pub title: String,
    pub state: TaskState,
    pub weight: Weight,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub age_seconds: u64,
}

impl TaskInfo {
    /// The protocol's task message. See `docs/protocol.md`.
    pub fn to_message(&self) -> serde_json::Value {
        let mut message = json!({
            "type": "task",
            "id": self.id,
            "state": self.state.as_str(),
            "title": self.title,
            "weight": self.weight.as_str(),
        });
        if let Some(detail) = &self.detail {
            message["detail"] = json!(detail);
        }
        message
    }
}

/// The signals one task can receive while it runs.
#[derive(Debug)]
struct Signals {
    stand_aside: AtomicBool,
    cancelled: AtomicBool,
}

/// A handle to a running task, held by whoever is doing the work.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    signals: Arc<Signals>,
}

impl Task {
    pub fn cancelled(&self) -> bool {
        self.signals.cancelled.load(Ordering::Acquire)
    }

    fn standing_aside(&self) -> bool {
        self.signals.stand_aside.load(Ordering::Acquire)
    }
}

/// Raised at a checkpoint when the task has been cancelled.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

struct Entry {
    info: TaskInfo,
    started: Instant,
    updated: Instant,
    signals: Arc<Signals>,
}

/// Everything in flight, and who has to wait for whom.
#[derive(Debug, Default)]
pub struct TaskManager {
    inner: Mutex<BTreeMap<String, Entry>>,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry").field("info", &self.info).finish()
    }
}

impl TaskManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&self, title: impl Into<String>, weight: Weight) -> Task {
        let id = format!("task-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let signals = Arc::new(Signals {
            stand_aside: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        });
        let now = Instant::now();
        let entry = Entry {
            info: TaskInfo {
                id: id.clone(),
                title: title.into(),
                state: TaskState::Queued,
                weight,
                detail: None,
                age_seconds: 0,
            },
            started: now,
            updated: now,
            signals: Arc::clone(&signals),
        };
        self.inner.lock().expect("tasks").insert(id.clone(), entry);
        Task { id, signals }
    }

    /// Update a task's state, returning the message to put on the wire.
    ///
    /// Returning rather than emitting keeps this module free of any opinion
    /// about transport, which is what lets it be tested without a server.
    pub fn set_state(
        &self,
        task: &Task,
        state: TaskState,
        detail: Option<&str>,
    ) -> Option<serde_json::Value> {
        let mut guard = self.inner.lock().expect("tasks");
        let entry = guard.get_mut(&task.id)?;
        entry.info.state = state;
        if let Some(detail) = detail {
            entry.info.detail = Some(detail.to_string());
        }
        entry.info.age_seconds = entry.started.elapsed().as_secs();
        entry.updated = Instant::now();
        Some(entry.info.to_message())
    }

    pub fn info(&self, id: &str) -> Option<TaskInfo> {
        self.inner
            .lock()
            .expect("tasks")
            .get(id)
            .map(|e| e.info.clone())
    }

    pub fn active(&self) -> Vec<TaskInfo> {
        let guard = self.inner.lock().expect("tasks");
        let mut tasks: Vec<TaskInfo> = guard
            .values()
            .filter(|entry| entry.info.state.is_active())
            .map(|entry| {
                let mut info = entry.info.clone();
                info.age_seconds = entry.started.elapsed().as_secs();
                info
            })
            .collect();
        tasks.sort_by_key(|info| std::cmp::Reverse(info.age_seconds));
        tasks
    }

    /// Whether a heavy task is currently occupying the machine.
    ///
    /// A *suspended* heavy task does not count: it has already stood aside,
    /// and asking it to do so again would be asking twice.
    pub fn heavy_running(&self) -> Vec<String> {
        self.inner
            .lock()
            .expect("tasks")
            .values()
            .filter(|entry| {
                entry.info.weight == Weight::Heavy && entry.info.state == TaskState::Running
            })
            .map(|entry| entry.info.id.clone())
            .collect()
    }

    /// Whether an incoming turn justifies asking heavy work to wait.
    ///
    /// The frontend has already decided a human is waiting; this is the
    /// backend's side of the same question — is anything actually in the way?
    pub fn should_preempt(&self, requested: bool) -> bool {
        requested && !self.heavy_running().is_empty()
    }

    /// Ask running heavy work to yield at its next checkpoint.
    pub fn stand_aside(&self, reason: &str) -> Vec<(String, serde_json::Value)> {
        let ids = self.heavy_running();
        let mut messages = Vec::new();
        let mut guard = self.inner.lock().expect("tasks");
        for id in ids {
            if let Some(entry) = guard.get_mut(&id) {
                entry.signals.stand_aside.store(true, Ordering::Release);
                entry.info.state = TaskState::Suspended;
                entry.info.detail = Some(reason.to_string());
                messages.push((id.clone(), entry.info.to_message()));
            }
        }
        messages
    }

    /// Let suspended work continue.
    pub fn resume(&self, ids: &[String]) -> Vec<serde_json::Value> {
        let mut guard = self.inner.lock().expect("tasks");
        let mut messages = Vec::new();
        for id in ids {
            if let Some(entry) = guard.get_mut(id) {
                if entry.info.state == TaskState::Cancelled {
                    continue;
                }
                entry.signals.stand_aside.store(false, Ordering::Release);
                entry.info.state = TaskState::Running;
                entry.info.detail = Some("picking that back up".into());
                messages.push(entry.info.to_message());
            }
        }
        messages
    }

    /// Call wherever stopping is free.
    ///
    /// Returns immediately unless the task has been asked to stand aside, in
    /// which case it waits until it may continue. `Err(Cancelled)` if the
    /// task was cancelled — the only thing that ends a task from outside.
    pub async fn checkpoint(&self, task: &Task) -> Result<(), Cancelled> {
        if task.cancelled() {
            return Err(Cancelled);
        }
        while task.standing_aside() {
            if task.cancelled() {
                return Err(Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if task.cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    /// Abandon work. `None` cancels everything in flight.
    pub fn cancel(&self, id: Option<&str>) -> Vec<(String, serde_json::Value)> {
        let mut guard = self.inner.lock().expect("tasks");
        let targets: Vec<String> = match id {
            Some(id) => vec![id.to_string()],
            None => guard
                .values()
                .filter(|entry| entry.info.state.is_active())
                .map(|entry| entry.info.id.clone())
                .collect(),
        };
        let mut messages = Vec::new();
        for id in targets {
            if let Some(entry) = guard.get_mut(&id) {
                if !entry.info.state.is_active() {
                    continue;
                }
                entry.signals.cancelled.store(true, Ordering::Release);
                // A cancelled task must not sit waiting at a checkpoint
                // forever, so release it as well as marking it.
                entry.signals.stand_aside.store(false, Ordering::Release);
                entry.info.state = TaskState::Cancelled;
                entry.info.detail = Some("cancelled".into());
                messages.push((id.clone(), entry.info.to_message()));
            }
        }
        messages
    }

    /// Drop finished tasks nobody will ask about again.
    pub fn prune(&self, keep_for: Duration) {
        let mut guard = self.inner.lock().expect("tasks");
        guard.retain(|_, entry| entry.info.state.is_active() || entry.updated.elapsed() < keep_for);
    }

    /// A sentence describing the current workload, for speaking aloud.
    pub fn spoken_summary(&self) -> String {
        let active = self.active();
        match active.len() {
            0 => "Nothing at the moment.".to_string(),
            1 => {
                let task = &active[0];
                let detail = task
                    .detail
                    .as_deref()
                    .map(|d| format!(" Right now: {d}."))
                    .unwrap_or_default();
                let suspended = if task.state == TaskState::Suspended {
                    " It's paused while I deal with this."
                } else {
                    ""
                };
                format!("I'm {}.{detail}{suspended}", task.title)
            }
            n => {
                let titles: Vec<&str> = active.iter().take(3).map(|t| t.title.as_str()).collect();
                format!("{n} things: {}.", titles.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_checkpoint_is_free_when_nothing_is_urgent() {
        let manager = TaskManager::new();
        let task = manager.create("building", Weight::Heavy);
        manager.set_state(&task, TaskState::Running, None);
        tokio::time::timeout(Duration::from_millis(500), manager.checkpoint(&task))
            .await
            .expect("did not return promptly")
            .expect("should not be cancelled");
    }

    #[tokio::test]
    async fn standing_aside_holds_at_the_checkpoint_then_resumes() {
        let manager = Arc::new(TaskManager::new());
        let task = manager.create("building", Weight::Heavy);
        manager.set_state(&task, TaskState::Running, None);

        let progress = Arc::new(Mutex::new(Vec::new()));
        let runner = {
            let manager = Arc::clone(&manager);
            let task = task.clone();
            let progress = Arc::clone(&progress);
            tokio::spawn(async move {
                for step in 0..3 {
                    manager.checkpoint(&task).await.unwrap();
                    progress.lock().unwrap().push(step);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
        };

        tokio::time::sleep(Duration::from_millis(10)).await;
        let suspended = manager.stand_aside("answering something short");
        assert_eq!(suspended.len(), 1);
        tokio::time::sleep(Duration::from_millis(150)).await;
        let held = progress.lock().unwrap().len();
        assert!(held < 3, "it should still be waiting");

        manager.resume(std::slice::from_ref(&task.id));
        tokio::time::timeout(Duration::from_secs(2), runner)
            .await
            .expect("did not finish")
            .unwrap();
        // It paused, then finished what it was doing: nothing was lost.
        assert_eq!(*progress.lock().unwrap(), vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn cancellation_ends_the_task_and_releases_a_held_checkpoint() {
        let manager = Arc::new(TaskManager::new());
        let task = manager.create("building", Weight::Heavy);
        manager.set_state(&task, TaskState::Running, None);
        manager.stand_aside("waiting");

        let runner = {
            let manager = Arc::clone(&manager);
            let task = task.clone();
            tokio::spawn(async move { manager.checkpoint(&task).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(manager.cancel(Some(&task.id)).len(), 1);

        let result = tokio::time::timeout(Duration::from_secs(2), runner)
            .await
            .expect("did not return")
            .unwrap();
        assert_eq!(result, Err(Cancelled));
        assert_eq!(manager.info(&task.id).unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn a_suspended_heavy_task_no_longer_occupies_the_machine() {
        let manager = TaskManager::new();
        let task = manager.create("building", Weight::Heavy);
        manager.set_state(&task, TaskState::Running, None);
        assert!(manager.should_preempt(true));
        manager.stand_aside("waiting");
        assert!(!manager.should_preempt(true));
        // Still active, though: it has not been abandoned.
        assert_eq!(manager.active().len(), 1);
    }

    #[test]
    fn preemption_is_only_requested_when_something_is_in_the_way() {
        let manager = TaskManager::new();
        assert!(!manager.should_preempt(true));
        let task = manager.create("building", Weight::Heavy);
        manager.set_state(&task, TaskState::Running, None);
        assert!(manager.should_preempt(true));
        assert!(!manager.should_preempt(false));
    }

    #[test]
    fn task_messages_match_the_protocol() {
        let manager = TaskManager::new();
        let task = manager.create("fixing the failing tests", Weight::Heavy);
        let message = manager
            .set_state(&task, TaskState::Running, Some("running the suite"))
            .unwrap();
        assert_eq!(message["type"], "task");
        assert_eq!(message["state"], "running");
        assert_eq!(message["weight"], "heavy");
        assert_eq!(message["detail"], "running the suite");
    }

    #[test]
    fn the_spoken_summary_reads_like_speech() {
        let manager = TaskManager::new();
        assert_eq!(manager.spoken_summary(), "Nothing at the moment.");
        let task = manager.create("fixing the failing tests", Weight::Heavy);
        manager.set_state(&task, TaskState::Running, Some("running the test suite"));
        let summary = manager.spoken_summary();
        assert!(
            summary.starts_with("I'm fixing the failing tests."),
            "{summary}"
        );
        assert!(summary.contains("running the test suite"));
    }

    #[test]
    fn finished_tasks_are_pruned_eventually() {
        let manager = TaskManager::new();
        let task = manager.create("done", Weight::Light);
        manager.set_state(&task, TaskState::Completed, None);
        manager.prune(Duration::from_secs(0));
        assert!(manager.active().is_empty());
    }
}
