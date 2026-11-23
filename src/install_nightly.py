#!/usr/bin/env python3
"""
install_nightly.py -- run the clipping batch automatically every day (macOS launchd).

    python3 install_nightly.py --install          # every day at 06:00 (default)
    python3 install_nightly.py --install --time 04:30
    python3 install_nightly.py --test             # run it once NOW the same way (to check)
    python3 install_nightly.py --status           # is it installed, when did it last run
    python3 install_nightly.py --uninstall        # stop the daily run

What the daily run does (nightly_run.sh, written next to this script):
  * loads your keys (~/.zshrc, ~/.twitch_keys) -- launchd doesn't load them by itself
  * skips if a batch is already running
  * caffeinate + batch_run.py streamers.txt --layout auto   (reads streamers.txt fresh)
  * refreshes the post queue (post_queue.py --no-open)
  * logs to nightly_logs/<date>.log
Nothing is uploaded or posted -- you still approve every clip on the review page.

The Mac must be on/awake at that time: plugged in, lid OPEN. If it's asleep, the run
starts as soon as it wakes. To wake it automatically a minute before (asks for your
Mac password once):   sudo pmset repeat wakeorpoweron MTWRFSU 05:59:00
"""
import argparse
import datetime
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = "com.chuzzzclips.nightly"
PLIST = os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")
SCRIPT = os.path.join(HERE, "nightly_run.sh")
PY = sys.executable

RUNNER = r'''#!/bin/zsh
# nightly_run.sh -- written by install_nightly.py. Runs the clipping batch once.
source ~/.zshrc >/dev/null 2>&1
[ -f ~/.twitch_keys ] && source ~/.twitch_keys
cd "__HERE__" || exit 1
mkdir -p nightly_logs
LOG="nightly_logs/$(date +%Y-%m-%d_%H%M).log"
{
  echo "=== nightly run $(date) ==="
  if pgrep -f "batch_run.py" >/dev/null; then
    echo "a batch is already running -- skipping tonight"; exit 0
  fi
  if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ANTHROPIC_API_KEY not found (is it exported in ~/.zshrc?) -- stopping"; exit 1
  fi
  /usr/bin/caffeinate -i "__PY__" -u batch_run.py streamers.txt --layout auto
  echo "=== batch finished $(date) ==="
  "__PY__" post_queue.py --no-open
  echo "=== done $(date) ==="
} >> "$LOG" 2>&1
'''

PLIST_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>__LABEL__</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>__SCRIPT__</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>__H__</integer><key>Minute</key><integer>__M__</integer></dict>
  <key>WorkingDirectory</key><string>__HERE__</string>
  <key>StandardOutPath</key><string>__HERE__/nightly_logs/launchd.out</string>
  <key>StandardErrorPath</key><string>__HERE__/nightly_logs/launchd.err</string>
</dict>
</plist>
'''


def launchctl(*a):
    return subprocess.run(["launchctl", *a], capture_output=True, text=True)


def write_runner():
    open(SCRIPT, "w").write(RUNNER.replace("__HERE__", HERE).replace("__PY__", PY))
    os.chmod(SCRIPT, 0o755)


def install(hhmm):
    h, m = (int(x) for x in hhmm.split(":"))
    if not (0 <= h < 24 and 0 <= m < 60):
        sys.exit("time must be HH:MM, e.g. 06:00")
    for f in ("batch_run.py", "streamers.txt", "post_queue.py"):
        if not os.path.exists(os.path.join(HERE, f)):
            sys.exit(f"ERROR: {f} not found next to this script -- put it in ~/Downloads. Nothing changed.")
    os.makedirs(os.path.join(HERE, "nightly_logs"), exist_ok=True)
    write_runner()
    os.makedirs(os.path.dirname(PLIST), exist_ok=True)
    xml = (PLIST_XML.replace("__LABEL__", LABEL).replace("__SCRIPT__", SCRIPT)
           .replace("__HERE__", HERE).replace("__H__", str(h)).replace("__M__", str(m)))
    open(PLIST, "w").write(xml)
    launchctl("unload", PLIST)
    r = launchctl("load", PLIST)
    if r.returncode != 0:
        sys.exit(f"launchctl load failed: {r.stderr.strip()}")
    print(f"installed: the batch will run every day at {h:02d}:{m:02d}")
    print(f"  runner: {SCRIPT}\n  logs:   {HERE}/nightly_logs/")
    print(f"  optional wake-up 1 min before (asks for your Mac password once):\n"
          f"    sudo pmset repeat wakeorpoweron MTWRFSU {h:02d}:{max(m - 1, 0):02d}:00")
    print("  Check it works now with:  python3 install_nightly.py --test")


def status():
    loaded = LABEL in launchctl("list").stdout
    print(f"installed: {'yes' if os.path.exists(PLIST) else 'no'}   loaded: {'yes' if loaded else 'no'}")
    if os.path.exists(PLIST):
        x = open(PLIST).read()
        try:
            h = x.split("<key>Hour</key><integer>")[1].split("<")[0]
            m = x.split("<key>Minute</key><integer>")[1].split("<")[0]
            print(f"runs daily at {int(h):02d}:{int(m):02d}")
        except Exception:
            pass
    logs = sorted(glob.glob(os.path.join(HERE, "nightly_logs", "20*.log")))
    if logs:
        print(f"last run log: {logs[-1]}")
        tail = open(logs[-1], errors="ignore").read().splitlines()[-6:]
        print("  " + "\n  ".join(tail))
    else:
        print("no runs yet")
    err = os.path.join(HERE, "nightly_logs", "launchd.err")
    if os.path.exists(err) and os.path.getsize(err):
        print("launchd errors (last lines):\n  " + "\n  ".join(open(err).read().splitlines()[-5:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--test", action="store_true", help="run once now through launchd")
    ap.add_argument("--time", default="06:00")
    a = ap.parse_args()
    if sys.platform != "darwin":
        sys.exit("macOS only.")
    if a.install:
        install(a.time)
    elif a.uninstall:
        launchctl("unload", PLIST)
        if os.path.exists(PLIST):
            os.remove(PLIST)
        print("uninstalled: no more daily runs (your files are untouched)")
    elif a.test:
        if not os.path.exists(PLIST):
            sys.exit("not installed yet -- run with --install first")
        r = launchctl("start", LABEL)
        print("started a run now (in the background). Watch it with:\n"
              f"  tail -f \"$(ls -t {HERE}/nightly_logs/20*.log | head -1)\"\n"
              "Stop it any time with:  pkill -f batch_run.py")
        if r.returncode != 0:
            print(r.stderr)
    else:
        status()


if __name__ == "__main__":
    main()
