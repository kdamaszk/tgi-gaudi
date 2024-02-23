use crate::infer::InferError;
use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::cmp::min;
use std::collections::VecDeque;
use text_generation_client::{Batch, Request};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
}

/// Request Queue
#[derive(Debug, Clone)]
pub(crate) struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
}

impl Queue {
    pub(crate) fn new(
        requires_padding: bool,
        max_input_length: u32,
        max_total_tokens: u32,
        block_size: u32,
        window_size: Option<u32>
    ) -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(queue_task(
            requires_padding,
            max_input_length,
            max_total_tokens,
            block_size,
            window_size,
            queue_receiver,
        ));

        Self { queue_sender }
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub(crate) fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::Append(Box::new(entry), Span::current()))
            .unwrap();
    }

    // Get the next batch
    #[instrument(skip(self))]
    pub(crate) async fn next_batch(
        &self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span: Span::current(),
            })
            .unwrap();
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.unwrap()
    }
}

// Background task responsible of the queue state
async fn queue_task(
    requires_padding: bool,
    max_input_length: u32,
    max_total_tokens: u32,
    block_size: u32,
    window_size: Option<u32>,
    mut receiver: mpsc::UnboundedReceiver<QueueCommand>,
) {
    // let smart_scheduling: bool = env::var("SMART_SCHEDULING").ok().map_or(false, |value| value.to_lowercase() == "true");

    let mut state = BucketizedState::new(
        requires_padding,
        max_input_length,
        max_total_tokens,
        block_size,
        window_size
    );

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            QueueCommand::Append(entry, span) => {
                span.in_scope(|| state.append(*entry));
                metrics::increment_gauge!("tgi_queue_size", 1.0);
            }
            QueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span,
            } => span.in_scope(|| {
                let next_batch = state.next_batch(min_size, prefill_token_budget, token_budget);
                response_sender.send(next_batch).unwrap();
                metrics::gauge!("tgi_queue_size", state.entries.len() as f64);
            }),
        }
    }
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Maximum input length, required for padding scenario
    max_input_length: u32,

    /// Maximum input and output length, required for padding scenario
    max_total_tokens: u32,

    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    window_size: Option<u32>,
}

impl State {
    fn new(
        requires_padding: bool,
        max_input_length: u32,
        max_total_tokens: u32,
        block_size: u32,
        window_size: Option<u32>
    ) -> Self {
        Self {
            entries: VecDeque::with_capacity(128),
            next_id: 0,
            next_batch_id: 0,
            requires_padding,
            max_input_length,
            max_total_tokens,
            block_size,
            window_size,
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push_back((self.next_id, entry));
        self.next_id += 1;
    }

    // Get the next batch
    fn next_batch(
        &mut self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        if self.entries.is_empty() {
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                return None;
            }
        }

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(self.entries.len());
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(self.entries.len(), BuildNoHashHasher::default());

        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;

        // Pop entries starting from the front of the queue
        while let Some((id, mut entry)) = self.entries.pop_front() {
            // Filter entries where the response receiver was dropped (== entries where the request
            // was dropped by the client)
            if entry.response_tx.is_closed() {
                metrics::increment_counter!("tgi_request_failure", "err" => "dropped");
                continue;
            }

            if self.requires_padding {
                // We pad to max input length in the Python shards
                // We need to take these padding tokens into the equation
                prefill_tokens = (batch_requests.len() + 1) as u32 * self.max_input_length;
            } else {
                // pad to block size
                prefill_tokens += ((entry.request.input_length + self.block_size - 1)
                    / self.block_size)
                    * self.block_size;
            }

            if self.requires_padding {
                // We pad to max total tokens in the Python shards
                // We need to take these padding tokens into the equation
                decode_tokens = (batch_requests.len() + 1) as u32 * (self.max_total_tokens - self.max_input_length);
            } else {
                let max_new_tokens = match self.window_size {
                    None => entry.request.stopping_parameters.max_new_tokens,
                    Some(window_size) => min(
                        window_size.saturating_sub(entry.request.input_length),
                        entry.request.stopping_parameters.max_new_tokens,
                    ),
                };

                // pad to block size
                decode_tokens +=
                    ((max_new_tokens + self.block_size - 1) / self.block_size) * self.block_size;
            }

            if prefill_tokens > prefill_token_budget
                || (prefill_tokens + decode_tokens) > token_budget
            {
                // Entry is over budget
                // Add it back to the front
                self.entries.push_front((id, entry));
                break;
            }

            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            batch_requests.push(Request {
                id,
                prefill_logprobs: entry.request.decoder_input_details,
                inputs: entry.request.inputs.clone(),
                truncate: entry.request.truncate,
                parameters: Some(entry.request.parameters.clone()),
                stopping_parameters: Some(entry.request.stopping_parameters.clone()),
                top_n_tokens: entry.request.top_n_tokens,
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
        }

        // Empty batch
        if batch_requests.is_empty() {
            return None;
        }

        // Check if our batch is big enough
        if let Some(min_size) = min_size {
            // Batch is too small
            if batch_requests.len() < min_size {
                // Add back entries to the queue in the correct order
                for r in batch_requests.into_iter().rev() {
                    let id = r.id;
                    let entry = batch_entries.remove(&id).unwrap();
                    self.entries.push_front((id, entry));
                }

                return None;
            }
        }

        // Final batch size
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
            max_tokens: (prefill_tokens + decode_tokens),
        };
        // Increment batch id
        self.next_batch_id += 1;

        metrics::histogram!("tgi_batch_next_size", batch.size as f64);

        Some((batch_entries, batch, next_batch_span))
    }
}

#[derive(Debug)]
struct Entries {
    time_limit: f32,
    fast_input_length: u32,
    fast_entries: VecDeque<(f32, (u64, Entry))>,
    slow_entries: VecDeque<(f32, (u64, Entry))>,
}

impl Entries {
    fn new(max_input_length: u32) -> Self {
        Self {
            time_limit: 2.0, // assume 2s limit per request
            fast_input_length: max_input_length / 4, // fast queries are smaller than 1/4 of max_input_length
            fast_entries: VecDeque::with_capacity(128),
            slow_entries: VecDeque::with_capacity(128)
        }
    }

    fn append(&mut self, id: u64, entry: Entry) {
        match entry.request.input_length <= self.fast_input_length {
            true => self.fast_entries.push_back((0.0, (id, entry))),
            false => self.slow_entries.push_back((0.0, (id, entry))),
        }
    }

    fn is_empty(&mut self) -> bool {
        self.fast_entries.is_empty() && self.slow_entries.is_empty()
    }

    fn len(&mut self) -> usize {
        self.fast_entries.len() + self.slow_entries.len()
    }

    fn next_batch(&mut self, min_size: Option<usize>, budget: usize) -> Vec<(u64, Entry)> {
        let fast_bs: usize = min(8, budget);
        let slow_bs: usize = min(4, budget);

        self.update_queues();

        let (bs, entries) = if self.batch_available(false, slow_bs) {
            (slow_bs, &mut self.slow_entries)
        } else if self.batch_available(true, fast_bs) {
            (fast_bs, &mut self.fast_entries)
        } else if min_size.is_none() && self.fast_entries.len() > 0 {
            (fast_bs, &mut self.fast_entries)
        } else if min_size.is_none() && self.slow_entries.len() > 0 {
            (slow_bs, &mut self.slow_entries)
        } else {
            (0, &mut self.fast_entries)
        };

        let mut batch: Vec<(u64, Entry)> = Vec::with_capacity(budget);
        while batch.len() < bs && entries.len() > 0 {
            let (_, (id, entry)) = entries.pop_front().unwrap();
            batch.push((id, entry));
        }
        batch
    }

    fn batch_available(&mut self, prefer_fast: bool, bs: usize) -> bool {
        let entries = match prefer_fast {
            true => &self.fast_entries,
            false => &self.slow_entries,
        };
        (
            // entire batch of punctual requests available
            entries.len() >= bs && entries[bs-1].0 >= 0.0
        ) || (
            // any request running out of time (300ms buffer)
            entries.len() > 0 && entries[0].0 > (self.time_limit - 0.3)
        )
    }

    fn update_queues(&mut self) {
        for entries in &mut [&mut self.fast_entries, &mut self.slow_entries] {
            for (time, (_, entry)) in entries.iter_mut() {
                // update time as a seconds since queuing
                *time = Instant::now().duration_since(entry.queue_time).as_secs_f32();

                // requests older than threshold should be moved to the end of the queue
                if *time >= (self.time_limit - 0.1) {
                    *time = f32::MIN;
                }
            }
            entries.make_contiguous().sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
        }
    }
}


/// Queue State
#[derive(Debug)]
struct BucketizedState {
    /// Queue entries organized in a Vec
    entries: Entries,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Maximum input length, required for padding scenario
    max_input_length: u32,

    /// Maximum input and output length, required for padding scenario
    max_total_tokens: u32,

    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    window_size: Option<u32>,
}

impl BucketizedState {
    fn new(
        requires_padding: bool,
        max_input_length: u32,
        max_total_tokens: u32,
        block_size: u32,
        window_size: Option<u32>
    ) -> Self {
        Self {
            entries: Entries::new(max_input_length),
            next_id: 0,
            next_batch_id: 0,
            requires_padding,
            max_input_length,
            max_total_tokens,
            block_size,
            window_size,
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        self.entries.append(self.next_id, entry);
        self.next_id += 1;
    }

    // Get the next batch
    fn next_batch(
        &mut self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        if self.entries.is_empty() {
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                return None;
            }
        }

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(self.entries.len());
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(self.entries.len(), BuildNoHashHasher::default());

        // Get batch based on the budget
        let budget: usize = min(
            prefill_token_budget / self.max_input_length, // space left for prefill
            token_budget / self.max_total_tokens // space left for decode
        ) as usize;
        let mut batch: Vec<(u64, Entry)> = self.entries.next_batch(min_size, budget);

        // Empty batch
        if batch.len() == 0 {
            return None;
        }

        while let Some((id, mut entry)) = batch.pop() {
            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            batch_requests.push(Request {
                id,
                prefill_logprobs: entry.request.decoder_input_details,
                inputs: entry.request.inputs.clone(),
                truncate: entry.request.truncate,
                parameters: Some(entry.request.parameters.clone()),
                stopping_parameters: Some(entry.request.stopping_parameters.clone()),
                top_n_tokens: entry.request.top_n_tokens,
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
        }

        // Final batch size
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
            max_tokens: self.max_total_tokens * size,
        };
        // Increment batch id
        self.next_batch_id += 1;

        metrics::histogram!("tgi_batch_next_size", batch.size as f64);

        Some((batch_entries, batch, next_batch_span))
    }
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    NextBatch {
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
    use tracing::info_span;

    fn _get_entry(input_length: u32) -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        let (response_tx, receiver_tx) = mpsc::unbounded_channel();

        let inputs = "test".repeat(input_length as usize).to_string();
        let entry = Entry {
            request: ValidGenerateRequest {
                inputs: inputs,
                input_length: input_length,
                truncate: 0,
                decoder_input_details: false,
                parameters: NextTokenChooserParameters {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 0.0,
                    typical_p: 0.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 0.0,
                    watermark: false,
                },
                stopping_parameters: StoppingCriteriaParameters {
                    ignore_eos_token: false,
                    max_new_tokens: 1,
                    stop_sequences: vec![],
                },
                top_n_tokens: 0,
            },
            response_tx,
            span: info_span!("entry"),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
        };
        (entry, receiver_tx)
    }

    fn default_entry() -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        _get_entry(0)
    }

    fn default_state() -> State {
        State::new(false, 100, 200, 1, None)
    }

    #[test]
    fn test_smart_state() {
        let max_input_length = 1024;
        let max_total_tokens = 2048;
        let mut state = BucketizedState::new(
            true,
            max_input_length,
            max_total_tokens,
            1,
            None
        );

        for _ in 0..10 {
            let (entry, _guard) = _get_entry(100);
            state.append(entry);
            let (entry, _guard) = _get_entry(500);
            state.append(entry);
        }

        let next_batch = state.next_batch(None, max_input_length*4, max_total_tokens*4);
        if let Some((entries, batch, _)) = next_batch {
            assert_eq!(batch.size, 4);
        }
    }

    #[test]
    fn test_smart_state_get_only_one() {
        let max_input_length = 1024;
        let max_total_tokens = 2048;
        let mut state = BucketizedState::new(
            true,
            max_input_length,
            max_total_tokens,
            1,
            None
        );

        let (entry, _guard) = _get_entry(100);
        state.append(entry);

        let next_batch = state.next_batch(Some(4), max_input_length*4, max_total_tokens*4);
        assert!(next_batch.is_none());

        let next_batch = state.next_batch(None, max_input_length*4, max_total_tokens*4);
        if let Some((entries, batch, _)) = next_batch {
            assert_eq!(batch.size, 1);
        }
    }


    #[test]
    fn test_append() {
        let mut state = default_state();
        let (entry, _guard) = default_entry();

        assert_eq!(state.next_id, 0);
        assert_eq!(state.entries.len(), 0);

        state.append(entry);

        assert_eq!(state.next_id, 1);
        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0).unwrap();
        assert_eq!(id, 0);
    }

    #[test]
    fn test_next_batch_empty() {
        let mut state = default_state();

        assert!(state.next_batch(None, 1, 1).is_none());
        assert!(state.next_batch(Some(1), 1, 1).is_none());
    }

    #[test]
    fn test_next_batch_min_size() {
        let mut state = default_state();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, 2, 2).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.append(entry3);

        assert!(state.next_batch(Some(2), 2, 2).is_none());

        assert_eq!(state.next_id, 3);
        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0).unwrap();
        assert_eq!(id, 2);
    }

    #[test]
    fn test_next_batch_token_budget() {
        let mut state = default_state();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, 1, 1).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.append(entry3);

        let (entries, batch, _) = state.next_batch(None, 3, 3).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);

        assert_eq!(state.next_id, 3);
        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 2);
    }

    #[tokio::test]
    async fn test_queue_append() {
        let queue = Queue::new(false, 100, 200, 1, None);
        let (entry, _guard) = default_entry();
        queue.append(entry);
    }

    #[tokio::test]
    async fn test_queue_next_batch_empty() {
        let queue = Queue::new(false, 100, 200, 1, None);

        assert!(queue.next_batch(None, 1, 1).await.is_none());
        assert!(queue.next_batch(Some(1), 1, 1).await.is_none());
    }

    #[tokio::test]
    async fn test_queue_next_batch_min_size() {
        let queue = Queue::new(false, 100, 200, 1, None);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, 2, 2).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3);

        // Not enough requests pending
        assert!(queue.next_batch(Some(2), 2, 2).await.is_none());
        // Not enough token budget
        assert!(queue.next_batch(Some(1), 0, 0).await.is_none());
        // Ok
        let (entries2, batch2, _) = queue.next_batch(Some(1), 2, 2).await.unwrap();
        assert_eq!(entries2.len(), 1);
        assert!(entries2.contains_key(&2));
        assert!(entries2.get(&2).unwrap().batch_time.is_some());
        assert_eq!(batch2.id, 1);
        assert_eq!(batch2.size, 1);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_budget() {
        let queue = Queue::new(false, 100, 200, 1, None);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, 1, 1).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3);

        let (entries, batch, _) = queue.next_batch(None, 3, 3).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_dropped_receiver() {
        let queue = Queue::new(false, 100, 200, 1, None);
        let (entry, _) = default_entry();
        queue.append(entry);

        assert!(queue.next_batch(None, 1, 1).await.is_none());
    }
}
