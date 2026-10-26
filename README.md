# ctxtax

See which Claude Code calls cost you the most, by context size.

I wrote this after pricing my own logs. About 30% of my cost came from the 11% of calls that sent 400k+ tokens of context. Sub-agents were about half. I run long sessions and lean on sub-agents a lot, so I'd expect most people's numbers to look quite different. This script shows yours.

```
$ python ctxtax.py
ctxtax: 19 calls, $1.95 at list-price equivalents

Context tax (by size of context sent per call)
  band         calls  % calls        cost  % cost
  <100k            3    15.8%        0.06    3.2%
  100-200k         3    15.8%        0.18    9.0%
  200-400k         7    36.8%        0.71   36.5%
  400k+            6    31.6%        1.00   51.2%

Sub-agents: $0.00 (0.0% of cost, 0 calls)
Errors: 0 API-error messages, 0 mention a limit; median gap after: n/a

What to try
  - 51% of cost is calls with 400k+ context. Start fresh sessions, or compact earlier.
```

(That output is from test data, not my logs.)

## Run it

Needs Python 3.11+. No packages to install.

```
curl -O https://raw.githubusercontent.com/TrueHorison/ctxtax/main/ctxtax.py
python ctxtax.py
```

It reads `~/.claude/projects` by default. Pass another folder as the first argument. It only reads Claude Code logs for now.

## Options

- `--json` prints the same numbers as JSON.
- `--share` prints a short summary with no paths or ids, if you want to paste it somewhere.
- `--meter` prints one line per 5-hour limit hit: the day, the usage in the 5 hours before it, and the model mix.

## How it works

Claude Code writes every API call to a `.jsonl` file, with token counts. ctxtax reads those counts, merges repeated lines for the same message, and prices each call at list price. A call's "context" is its input plus cache reads plus cache writes. Each call re-sends the whole context, so long sessions get expensive even with cheap cache reads.

## Prices

The `PRICES` dict at the top of `ctxtax.py` holds Anthropic's list prices, last checked 2026-10-26. Your real bill differs if you're on a plan or get a discount. Models it doesn't know use a default price and print a warning. Edit the dict to change any of this.

## Privacy

It only reads local files. There's no network code, and it never prints message text.

## Tests

```
python -m unittest discover tests
```

## License

MIT
