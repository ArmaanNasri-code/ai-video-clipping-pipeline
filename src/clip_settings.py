#!/usr/bin/env python3
"""
clip_settings.py -- turn finishing features on or off for all future clips.

    python3 clip_settings.py                  # show current settings
    python3 clip_settings.py watermark off
    python3 clip_settings.py captions on hook off

Features: captions, hook (the title at the top), watermark,
          steady_crop (v4 crop renderer; off = old v3 crop),
          gameplay_layout stack|wide (facecam over gameplay: stacked, or thin full-frame band).
Split screen is chosen per batch:  batch_run.py ... --layout full | auto | split
"""
import json
import os
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_settings.json")
DEFAULTS = {"captions": True, "hook": True, "watermark": True, "steady_crop": True,
            "gameplay_layout": "stack"}
CHOICES = {"gameplay_layout": ("stack", "wide")}


def main():
    try:
        s = {**DEFAULTS, **json.load(open(PATH))}
    except Exception:
        s = dict(DEFAULTS)
    args = [a.lower() for a in sys.argv[1:]]
    if len(args) % 2:
        sys.exit(__doc__)
    for k, v in zip(args[::2], args[1::2]):
        if k in CHOICES:
            if v not in CHOICES[k]:
                sys.exit(f"{k} can be: {' / '.join(CHOICES[k])}")
            s[k] = v
            continue
        if k not in DEFAULTS or v not in ("on", "off"):
            sys.exit(f"don't understand '{k} {v}'.\n{__doc__}")
        s[k] = v == "on"
    if args:
        json.dump(s, open(PATH, "w"), indent=2)
    for k, v in s.items():
        print(f"  {k:15s} {v if isinstance(v, str) else ('ON' if v else 'OFF')}")
    print("(applies to clips finished from now on)" if args else "")


if __name__ == "__main__":
    main()
