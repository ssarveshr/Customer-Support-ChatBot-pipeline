# Decision Log

- Use one brand/account for the first release to keep retrieval evidence and escalation policy unambiguous.
- Normalize inbound and outbound tweets into one customer-message/brand-response record.
- Pair Kaggle conversations through `response_tweet_id` because the source does not provide ready-made conversation rows.
- Keep the dataset size configurable so local development does not require indexing millions of tweets.
- Use Sentence-Transformers with ChromaDB so the embedding and storage layers can be replaced independently.
- Use Ollama behind a small HTTP client so a hosted model can replace it later.
- Start intent classification with explicit keyword rules to establish an interpretable baseline.
- Include a majority classifier to distinguish useful signal from a trivial class-frequency strategy.
- Escalate when confidence, evidence similarity, sensitivity, or generation grounding is insufficient.
- Treat retrieved historical responses as evidence, not as instructions to copy blindly.
- Keep the golden set separate from training/indexing data to reduce evaluation leakage.
- Require manual labels before the evaluation command writes metrics.
- Use an LLM judge for scalable reply review, but measure it against human scores before trusting it.
- Avoid fabricating headline metrics when the dataset, brand, or labels are not present.