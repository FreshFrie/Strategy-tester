# Asia Killzone → London Backtest Strategy (UTC-4 Data)

## Session Definitions (UTC-4)
| Session       | Time Range (UTC-4) | Purpose |
|---------------|-------------------|---------|
| Asia Curr     | `17:00` – `23:00` | Main Asia session for this trading day |
| Asia Killzone | `19:30` – `21:30` | Key level generation: record high/low and mark if extremes occurred here |
| Pre-London    | `21:00` – `23:00` | Optional early filter (skip if KZ levels broken too early) |
| London        | `23:00` – `05:00` | General session window |
London KZ '02:00 - 04:00'
| Takeout       | Configurable (e.g., `01:00` – `08:00`) | Trade execution window |

---

## Key Logic

1. **Asia Session Extremes**  
   - Combine `Asia Prev` + `Asia Curr` to find **session high** and **session low**.
   - Mark if the high or low was made **inside** the Asia Killzone.

2. **Bias Selection**  
   - **Only high in KZ** → Long toward that high.  
   - **Only low in KZ** → Short toward that low.  
   - **Both extremes in KZ** → Wait for first takeout in London then target the opposing level

3. **Entry Trigger**  
   - Based on **Pre-London** range:
     - `break` = enter on first close beyond range.
     - `retest` = break, then retest, then enter.

4. **Stops & Targets**  
   - Target = KZ level from bias logic.
   - Stop = distance from entry to maintain RR ratio (e.g., 1:1.5).

5. **Exit**  
   - TP, SL, or end of takeout window.

---
