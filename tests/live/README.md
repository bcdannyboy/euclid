# Euclid Live API Tests

Live tests are disabled unless `EUCLID_LIVE_API_TESTS=1` is set.

Strict release mode uses:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 ./scripts/live_api_smoke.sh
```

Required `.env` names:

- `FMP_API_KEY`
- `OPENAI_API_KEY`

The tests write sanitized evidence only. They must not print or artifact secret
values.
