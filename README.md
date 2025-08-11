# Strategy-tester

NAS100 Killzone Backtester with enhanced sizing modes and multi-entry capability.

## Features

- **Sizing Modes**: MNQ/NQ contracts or CFD fractional sizing
- **Multi-entries**: Multiple trades per day after killzone ends
- **Vectorized Performance**: Fast numpy-based first-hit detection
- **CLI Configuration**: Flexible command-line options
- **Timezone Handling**: Proper UTC→America/New_York conversion

## Usage

### Basic Usage (MNQ Micro Futures)
```bash
python main.py --csv NQ_5Years_8_11_2024.csv --engine pyarrow --sizing contracts --risk-pct 0.01 --multi-entries on
```

### CFD Fractional Sizing
```bash
python main.py --sizing cfd --point-value 1.0 --multi-entries on
```

### Single Entry Mode
```bash
python main.py --multi-entries off
```

### CLI Options

- `--csv PATH`: Path to CSV file (default: NQ_5Years_8_11_2024.csv)
- `--engine {pandas,pyarrow}`: CSV read engine
- `--sizing {contracts,cfd}`: Position sizing mode
- `--tick-value FLOAT`: $ per tick for contracts mode (default: 0.5 for MNQ)
- `--point-value FLOAT`: $ per point for CFD mode (default: 1.0)
- `--risk-pct FLOAT`: Risk percentage per trade (default: 0.01)
- `--multi-entries {on,off}`: Allow multiple entries per day (default: on)
- `--cutoff HH:MM`: Time cutoff in ET (default: 12:00)

## Data Format

CSV with columns: `Time,Open,High,Low,Close` where Time is in UTC format like "8/11/2019 23:05".

## Sessions & Killzones

- **Sessions** (America/New_York timezone):
  - London: 00:00 - 06:00 ET
  - Asia: 18:00 - 00:00 ET (previous day)

- **Killzones**:
  - London: 02:00 - 05:00 ET
  - Asia: 20:00 - 22:00 ET (previous day)

## Output

- `trades.csv`: Detailed trade log
- `equity_curve.csv`: Daily equity progression
