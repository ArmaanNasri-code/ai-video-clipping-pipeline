"""
TwitchAdapter: implements StreamPlatformAdapter for Twitch VODs.

IMPORTANT HONESTY NOTE: both get_vod_manifest() and get_chat_log() here
use Twitch's UNOFFICIAL, reverse-engineered GraphQL API (gql.twitch.tv),
not the public Helix API. This is the same category of thing as Kick's
Chat Replay API you're already using via cloudscraper -- undocumented,
not officially supported, could change without notice. Treat this the
same way: works today, verify it still works before relying on it in
production, and don't be surprised if Twitch changes something and this
needs a patch later.

get_stream_start_time() DOES use the official Helix API, since Twitch's
official /videos endpoint returns created_at directly -- this one
requires you to register a free Twitch developer app (Client-ID +
Client-Secret) at https://dev.twitch.tv/console/apps to get an app
access token. That's a one-time setup step, documented below.
"""

import json
from datetime import datetime, timezone
from platform_adapter_base import StreamPlatformAdapter

# This is the public Client-ID Twitch's own web player uses. It's widely
# known/used in open-source Twitch tooling (not a secret, not something
# you need to register) -- required as a header on unofficial GQL calls.
TWITCH_WEB_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

GQL_URL = "https://gql.twitch.tv/gql"


class TwitchAdapter(StreamPlatformAdapter):
    def __init__(self, helix_client_id: str = None, helix_client_secret: str = None):
        """
        helix_client_id / helix_client_secret: from your own registered
        Twitch dev app (https://dev.twitch.tv/console/apps), needed only
        for get_stream_start_time() via the official Helix API. Not
        needed for get_vod_manifest() or get_chat_log(), which use the
        unofficial GQL endpoint with the public web client-id instead.
        """
        self._helix_client_id = helix_client_id
        self._helix_client_secret = helix_client_secret
        self._helix_token = None

    def _get_helix_app_token(self):
        """Fetches an app access token via OAuth client-credentials flow.
        Cached for the life of this adapter instance."""
        import requests

        if self._helix_token:
            return self._helix_token
        if not self._helix_client_id or not self._helix_client_secret:
            raise ValueError(
                "get_stream_start_time() needs helix_client_id/helix_client_secret. "
                "Register a free app at https://dev.twitch.tv/console/apps and pass "
                "those credentials into TwitchAdapter(...)."
            )
        resp = requests.post(
            "https://id.twitch.tv/oauth2/token",
            params={
                "client_id": self._helix_client_id,
                "client_secret": self._helix_client_secret,
                "grant_type": "client_credentials",
            },
            timeout=15,
        )
        resp.raise_for_status()
        self._helix_token = resp.json()["access_token"]
        return self._helix_token

    def get_stream_start_time(self, stream_id: str) -> datetime:
        """
        stream_id here is the Twitch VOD ID (numeric, found in the VOD
        URL: twitch.tv/videos/<VOD_ID>).

        Uses the official Helix /videos endpoint to get created_at.
        Returns a UTC-aware datetime.
        """
        import requests

        token = self._get_helix_app_token()
        resp = requests.get(
            "https://api.twitch.tv/helix/videos",
            headers={
                "Client-Id": self._helix_client_id,
                "Authorization": f"Bearer {token}",
            },
            params={"id": stream_id},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        if not data:
            raise ValueError(f"No VOD found for id={stream_id}")
        created_at = data[0]["created_at"]  # e.g. "2026-08-08T04:13:31Z"
        return datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    def get_vod_manifest(self, stream_id: str) -> str:
        """
        stream_id here is the Twitch VOD ID.

        Real implementation via the unofficial PlaybackAccessToken GQL
        query (confirmed current query shape as of this writing --
        verify it still matches if this starts failing, since Twitch
        can change the schema without notice).

        Downloads the manifest locally and returns the local file path,
        same contract as KickAdapter.get_vod_manifest() will eventually
        have once that's automated.
        """
        import requests

        query = """
        query PlaybackAccessToken_Template($login: String!, $isLive: Boolean!, $vodID: ID!, $isVod: Boolean!, $playerType: String!) {
            streamPlaybackAccessToken(channelName: $login, params: {platform: "web", playerBackend: "mediaplayer", playerType: $playerType}) @include(if: $isLive) {
                value
                signature
                __typename
            }
            videoPlaybackAccessToken(id: $vodID, params: {platform: "web", playerBackend: "mediaplayer", playerType: $playerType}) @include(if: $isVod) {
                value
                signature
                __typename
            }
        }
        """
        variables = {
            "login": "",
            "isLive": False,
            "vodID": stream_id,
            "isVod": True,
            "playerType": "site",
        }

        resp = requests.post(
            GQL_URL,
            headers={"Client-Id": TWITCH_WEB_CLIENT_ID},
            json={"query": query, "variables": variables},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()

        token_data = payload.get("data", {}).get("videoPlaybackAccessToken")
        if not token_data:
            raise RuntimeError(
                f"No videoPlaybackAccessToken in response -- VOD may be private/expired, "
                f"or Twitch changed the query shape. Raw response: {json.dumps(payload)[:500]}"
            )

        token = token_data["value"]
        sig = token_data["signature"]

        manifest_url = (
            f"https://usher.ttvnw.net/vod/{stream_id}.m3u8"
            f"?nauth={token}&nauthsig={sig}&allow_source=true&player=twitchweb"
        )

        manifest_resp = requests.get(manifest_url, timeout=15)
        manifest_resp.raise_for_status()

        local_path = f"twitch_vod_{stream_id}_master.m3u8"
        with open(local_path, "w") as f:
            f.write(manifest_resp.text)
        self._last_manifest_base_url = manifest_url

        print(f"Saved Twitch master manifest to {local_path}")
        print("NOTE: like Kick, this is a MASTER manifest listing quality variants --")
        print("you'll likely need to fetch the specific quality sub-manifest URL from")
        print("inside this file before it can be used with fetch_segment_window().")

        return local_path

    def get_quality_submanifest(self, master_manifest_path: str, quality: str = "1080p") -> str:
        """
        Given a local MASTER manifest path (from get_vod_manifest()), find the
        stream variant matching `quality`, download its actual segment-level
        sub-manifest (the one with #EXT-X-PROGRAM-DATE-TIME tags), save it
        locally, and set self._last_manifest_base_url so fetch_segment_window()
        can correctly resolve Twitch's relative segment paths (e.g. "0.ts")
        against the right CDN base URL.

        `quality` matches against RESOLUTION in the #EXT-X-STREAM-INF line
        (e.g. "1080p" matches RESOLUTION=1920x1080, "720p" matches 1280x720).
        Common values: 1080p, 720p, 480p, 360p, 160p. Also accepts "chunked"
        to just grab the first/highest variant regardless of resolution.

        Returns the local path to the saved sub-manifest.
        """
        import requests as _requests

        quality_res_map = {
            "1080p": "1920x1080",
            "720p": "1280x720",
            "480p": "852x480",
            "360p": "640x360",
            "160p": "284x160",
        }
        target_res = quality_res_map.get(quality)

        with open(master_manifest_path) as f:
            lines = f.read().splitlines()

        sub_url = None
        for i, line in enumerate(lines):
            if not line.startswith("#EXT-X-STREAM-INF"):
                continue
            if quality == "chunked" or (target_res and f"RESOLUTION={target_res}" in line):
                sub_url = lines[i + 1]
                break

        if not sub_url:
            available = [l for l in lines if l.startswith("#EXT-X-STREAM-INF")]
            raise ValueError(
                f"Could not find quality '{quality}' in master manifest. "
                f"Available variants:\n" + "\n".join(available)
            )

        resp = _requests.get(sub_url, timeout=15)
        resp.raise_for_status()

        local_path = master_manifest_path.replace("_master.m3u8", f"_{quality}_sub.m3u8")
        with open(local_path, "w") as f:
            f.write(resp.text)

        self._last_manifest_base_url = sub_url

        print(f"Saved {quality} sub-manifest to {local_path} ({len(resp.text)} bytes)")
        return local_path

    def get_chat_log(self, stream_id: str, start_time: datetime, end_time: datetime,
                      output_file: str = None) -> list:
        """
        Uses TwitchDownloaderCLI (https://github.com/lay295/TwitchDownloader) as a
        subprocess instead of hitting Twitch's unofficial GQL API directly.

        WHY: the raw GQL VideoCommentsByOffsetOrCursor approach works for the first
        batch (by contentOffsetSeconds) but fails on subsequent cursor-based pages
        with an "IntegrityCheckFailed" error -- Twitch requires a Client-Integrity
        token (generated via in-browser device attestation) for continued pagination
        that we don't have a way to produce ourselves. TwitchDownloaderCLI is an
        actively maintained tool that handles this internally and reliably pulls
        full VOD chat histories (verified: 57,911 messages across a 7+ hour VOD in
        testing, vs. 58 messages before hitting the wall with raw GQL).

        Requires TwitchDownloaderCLI binary. Set self._cli_path or place it at
        ./twitchdownloader/TwitchDownloaderCLI relative to cwd.

        start_time/end_time currently unused for filtering -- TwitchDownloaderCLI
        exports the full VOD chat; filter the returned list by offset afterward
        if you only want a sub-range.
        """
        import subprocess
        import json
        import os
        import tempfile

        cli_path = getattr(self, "_cli_path", "./twitchdownloader/TwitchDownloaderCLI")
        if not os.path.exists(cli_path):
            raise FileNotFoundError(
                f"TwitchDownloaderCLI not found at {cli_path}. Download it from "
                f"https://github.com/lay295/TwitchDownloader/releases and set "
                f"adapter._cli_path, or place it at the default relative path."
            )

        tmp_path = os.path.join(tempfile.gettempdir(), f"twitch_chat_{stream_id}_{os.getpid()}.json")

        result = subprocess.run(
            [cli_path, "chatdownload", "--id", stream_id, "-o", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            raise RuntimeError(f"TwitchDownloaderCLI chatdownload failed: {result.stderr}")

        with open(tmp_path) as f:
            data = json.load(f)
        os.unlink(tmp_path)

        comments = data.get("comments", [])
        results = []
        lines = []
        for c in comments:
            offset = c.get("content_offset_seconds")
            username = c.get("commenter", {}).get("display_name", "unknown")
            text = c.get("message", {}).get("body", "")
            results.append((offset, username, text))
            lines.append(f"{offset} {username}: {text}\n")

        print(f"Fetched {len(results)} messages via TwitchDownloaderCLI")

        if output_file:
            with open(output_file, "w") as f:
                f.writelines(lines)
            print(f"Wrote {output_file}")

        return results

    def fetch_segment_window(self, manifest_path: str, target_time: datetime, window_seconds: int) -> str:
        """
        Twitch VOD segments are standard HLS (same #EXT-X-PROGRAM-DATE-TIME
        tag structure as Kick), so this should be reusable close to as-is
        from KickAdapter.fetch_segment_window() -- same manifest parsing
        and ffmpeg concat logic, no Twitch-specific changes needed at this
        layer once you have a valid local manifest.

        Not duplicating that logic here -- once you're ready to test this
        for real, this method can just delegate to the same parse/concat
        code as KickAdapter, since HLS segment mechanics don't differ by
        platform.
        """
        """
        Real implementation, ported from KickAdapter.fetch_segment_window().
        HLS segment/PROGRAM-DATE-TIME structure is standard across platforms,
        so the manifest-parsing and closest-segment logic is identical.

        DIFFERENCE FROM KICK: Twitch's CDN segment URLs from the manifest are
        typically pre-signed/self-contained, so downloads do NOT need Kick's
        --referer header trick. If downloads start failing with 403s, that
        assumption may need revisiting.

        target_time is expected as an already-UTC-aware datetime (unlike Kick's
        version, which assumes naive local EDT and converts internally) -- this
        adapter's get_stream_start_time() already returns UTC-aware directly
        from Helix's created_at, so no extra conversion belongs here.

        Returns the local path to the concatenated output mp4 (twitch_verify.mp4).
        """
        import re as _re
        import glob
        import os
        import subprocess as _subprocess

        def parse_manifest(path):
            from urllib.parse import urljoin
            base_url = getattr(self, "_last_manifest_base_url", None)
            with open(path, "r") as f:
                lines = f.readlines()
            entries = []
            pending_dt = None
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#EXTINF"):
                    continue
                m = _re.match(r"#EXT-X-PROGRAM-DATE-TIME:(.+)", line)
                if m:
                    ts = m.group(1)
                    if ts.endswith("Z"):
                        ts = ts[:-1] + "+00:00"
                    pending_dt = datetime.fromisoformat(ts)
                    continue
                if line.startswith("#"):
                    continue
                if pending_dt is not None:
                    # Twitch sub-manifests use relative segment paths (e.g. "0.ts")
                    # unlike Kick's full https:// URLs -- resolve against base_url.
                    url = line if line.startswith("http") else urljoin(base_url or "", line)
                    entries.append((url, pending_dt))
                    pending_dt = None
            if not entries:
                raise ValueError(
                    "No PROGRAM-DATE-TIME + URL pairs found in manifest. "
                    "Check that the manifest actually contains segment-level tags."
                )
            return entries

        def find_closest_segment(entries, target_utc):
            best_idx = 0
            best_diff = None
            for i, (_, dt) in enumerate(entries):
                diff = abs((dt - target_utc).total_seconds())
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_idx = i
            return best_idx, best_diff

        target_utc = target_time  # already UTC-aware, no conversion needed

        entries = parse_manifest(manifest_path)
        print(f"Parsed {len(entries)} segments from manifest.")
        print(f"Manifest covers: {entries[0][1].isoformat()} -> {entries[-1][1].isoformat()}")

        if target_utc < entries[0][1] or target_utc > entries[-1][1]:
            print("*** WARNING: target_utc falls OUTSIDE the manifest's covered range. ***")
            print("Double check you're using the manifest for the correct VOD.")

        idx, diff = find_closest_segment(entries, target_utc)
        seg_url, seg_dt = entries[idx]
        print(f"\nClosest segment: index {idx}")
        print(f"  URL: {seg_url}")
        print(f"  PROGRAM-DATE-TIME: {seg_dt.isoformat()}")
        print(f"  Difference from target: {diff:.3f} seconds")

        if len(entries) > 1:
            seg_duration = (entries[1][1] - entries[0][1]).total_seconds()
        else:
            seg_duration = 10.0
        segs_each_side = max(1, int(window_seconds / seg_duration))

        lo = max(0, idx - segs_each_side)
        hi = min(len(entries) - 1, idx + segs_each_side)
        window_entries = entries[lo:hi + 1]

        print(f"\nDownloading segments {lo} to {hi} ({len(window_entries)} segments, "
              f"~{len(window_entries)*seg_duration:.0f}s of video)...")

        # Clean up leftover .ts segments from prior runs before starting --
        # these accumulate fast and aren't auto-cleaned, causing disk-full errors.
        leftover = glob.glob("twitch_seg_*.ts")
        if leftover:
            print(f"Cleaning up {len(leftover)} leftover .ts files from prior runs...")
            for f in leftover:
                try:
                    os.remove(f)
                except OSError:
                    pass

        concat_list = []
        for i, (url, dt) in enumerate(window_entries):
            out_name = f"twitch_seg_{lo + i}.ts"
            result = _subprocess.run(
                ["curl", "-s", url, "-o", out_name],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  FAILED to download {url}: {result.stderr}")
                continue
            concat_list.append(out_name)
            print(f"  [{i+1}/{len(window_entries)}] {out_name}  (PDT: {dt.isoformat()})")

        if not concat_list:
            raise RuntimeError("No segments downloaded successfully.")

        with open("twitch_concat.txt", "w") as f:
            for fname in concat_list:
                f.write(f"file '{fname}'\n")

        out_mp4 = "twitch_verify.mp4"
        _subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "twitch_concat.txt",
             "-c", "copy", out_mp4],
            capture_output=True, text=True
        )

        print(f"\nDone. Output: {out_mp4}")
        print(f"This clip starts at real video timestamp: {window_entries[0][1].isoformat()}")
        print(f"Your target moment ({target_utc.isoformat()}) should appear at approximately "
              f"{(target_utc - window_entries[0][1]).total_seconds():.0f} seconds into this clip.")

        return out_mp4
