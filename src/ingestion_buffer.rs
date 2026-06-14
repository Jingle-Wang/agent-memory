use crate::models::Event;

/// Sliding-window ingestion buffer for multi-turn contextual memory extraction
/// (P0-2: fixes the 50% gold-evidence extraction bottleneck).
///
/// Collects consecutive conversation events (turns) and flushes them as a
/// single combined event when the configured window size is reached. This
/// gives the `MemoryExtractor` enough surrounding context to extract rich
/// facts — entity relationships, implied information, emotional arcs —
/// instead of thin single-turn snippets.
///
/// # Usage
///
/// ```ignore
/// let mut buf = IngestionBuffer::new(5);
/// for event in conversation {
///     if let Some(combined) = buf.push(event) {
///         // Extract memories from the combined multi-turn context
///         extractor.extract(&combined, None);
///     }
/// }
/// // Don't forget trailing turns
/// if let Some(combined) = buf.flush() {
///     extractor.extract(&combined, None);
/// }
/// ```
pub struct IngestionBuffer {
    events: Vec<Event>,
    window_size: usize,
    stride: usize,
}

impl IngestionBuffer {
    /// Create a new buffer with the given window size.
    ///
    /// `window_size` is the number of consecutive turns to collect before
    /// flushing. Typical values are 3–5 (LoCoMo conversations average 10–15
    /// turns per session, so a window of 5 gives roughly 3 extraction calls
    /// per session).
    pub fn new(window_size: usize) -> Self {
        assert!(
            window_size > 0,
            "IngestionBuffer window_size must be positive"
        );
        let stride = ((window_size + 1) / 2).max(1);
        Self {
            events: Vec::with_capacity(window_size),
            window_size,
            stride,
        }
    }

    /// Push an event into the buffer.
    ///
    /// Returns `Some(combined_event)` when the buffer reaches `window_size`
    /// — the combined event should be passed to the extractor. Returns `None`
    /// if the buffer is not yet full.
    pub fn push(&mut self, event: Event) -> Option<Event> {
        self.events.push(event);
        if self.events.len() >= self.window_size {
            Some(self.take_combined())
        } else {
            None
        }
    }

    /// Number of events currently held in the buffer.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Flush any remaining events as a combined event.
    ///
    /// Returns `None` if the buffer is already empty. Call this at the end
    /// of a conversation so the last partial window is not lost.
    pub fn flush(&mut self) -> Option<Event> {
        if self.events.is_empty() {
            return None;
        }
        Some(self.take_combined())
    }

    /// Drain: if the buffer is full, return a combined event and clear.
    /// Otherwise return `None` and leave the buffer unchanged.
    pub fn drain_if_full(&mut self) -> Option<Event> {
        if self.events.len() >= self.window_size {
            Some(self.take_combined())
        } else {
            None
        }
    }

    // ── internals ────────────────────────────────────────────────────────

    /// Build a combined event from all buffered events, then drain stride events
    /// keeping the last (window_size - stride) for overlap context.
    fn take_combined(&mut self) -> Event {
        let combined = self.build_combined_event();
        // Keep last (window_size - stride) events for overlap context
        if self.events.len() > self.stride {
            self.events.drain(..self.stride);
        } else {
            self.events.clear();
        }
        combined
    }

    fn build_combined_event(&self) -> Event {
        let mut text = format!("[Multi-turn conversation — {} turns]\n", self.events.len());
        for (i, event) in self.events.iter().enumerate() {
            text.push_str(&format!(
                "Turn {} — {}: {}\n",
                i + 1,
                event.actor,
                event.text
            ));
        }
        let namespace = self.events[0].namespace.clone();
        Event::new(text.trim().to_string())
            .namespace(namespace)
            .actor("multi-turn".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_flushes_when_full() {
        let mut buf = IngestionBuffer::new(3);
        assert!(buf.push(Event::new("Hello")).is_none());
        assert!(buf.push(Event::new("How are you?")).is_none());
        let combined = buf.push(Event::new("I'm good"));
        assert!(combined.is_some());
        // window_size=3, stride=2, so 1 overlap event remains
        assert_eq!(buf.len(), 3 - 2);
    }

    #[test]
    fn flush_drains_remaining() {
        let mut buf = IngestionBuffer::new(3);
        buf.push(Event::new("msg1"));
        buf.push(Event::new("msg2"));
        assert_eq!(buf.len(), 2);
        let combined = buf.flush();
        assert!(combined.is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn empty_flush_returns_none() {
        let mut buf = IngestionBuffer::new(3);
        assert!(buf.flush().is_none());
    }

    #[test]
    fn combined_event_contains_all_turns() {
        let mut buf = IngestionBuffer::new(2);
        buf.push(Event::new("Hello").actor("Alice").namespace("chat"));
        let combined = buf
            .push(Event::new("Hi!").actor("Bob").namespace("chat"))
            .unwrap();
        assert!(combined.text.contains("Alice"));
        assert!(combined.text.contains("Bob"));
        assert!(combined.text.contains("Hello"));
        assert!(combined.text.contains("Hi!"));
        assert_eq!(combined.namespace, "chat");
    }
}
