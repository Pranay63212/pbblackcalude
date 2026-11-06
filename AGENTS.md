# AGENTS.md

Rules for anyone (human or AI agent) working in this repository.

1. Run `python tests.py` before every commit. All tests must pass. Never delete or weaken a test to make it pass.
2. Strategy code may only use data up to and including the current bar's close. No lookahead.
3. Never lower the default fees or slippage in `fbt/costs.py`.
4. Never commit API keys, account numbers, or data files. Secrets come from environment variables only.
5. Any code that sends orders must go through the risk manager (once it exists).
6. Live trading code must default to paper mode. Live mode requires an explicit `--live` flag AND the environment variable `LIVE_TRADING_CONFIRMED=yes`.
7. Explain any change to order-sending or risk code in the commit message.

## Strategy

The trading strategy to implement is specified in `docs/STRATEGY.md`.
