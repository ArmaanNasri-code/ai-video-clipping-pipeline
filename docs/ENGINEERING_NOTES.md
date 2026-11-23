# Engineering notes

Problems that came up while building and running the pipeline, what caused them, and how each was resolved. Each fix was verified on real output before being kept.

---

### 1. Measuring hype from chat: unique chatters, not messages
**Problem.** Raw chat message rate is dominated by a few fast typers, emote spam and bots, so "spikes" were mostly noise.
**Fix.** Count *unique active chatters* per 15-second bucket and compare against a local median baseline (±300 s window, minimum baseline of 10). A real moment makes many different people react at once.

### 2. One transcription error hid the best moment of a stream
**Problem.** The most viral moment of one stream was invisible to every automatic signal. The speech model transcribed a guest's name ("Neon") as "young Navi", so the transcript scorer saw nothing interesting.
**Lesson.** Any single signal has blind spots. Detection now combines four independent sources (transcript, chat, a visual scan, and clips viewers made), and transcription accepts known names as hints.

### 3. Measuring recall with ground truth from viewers
**Problem.** "The stream was dry" and "we missed the good parts" look identical from inside the pipeline.
**Fix.** Use the clips Twitch viewers made (with their view counts and timestamps) as ground truth. For each top viewer clip, classify the outcome as *found & approved*, *found & rejected*, or *never found*. First result: 72 % of top viewer moments were never found. Adding viewer clips as a fourth candidate source cut that to 37 %.

### 4. The judge was rejecting popular moments for the wrong reasons
**Problem.** Popular talking and reaction moments scored 3–6 with reasons like "talking head, no visual story" and "too much dead time".
**Diagnosis.** The prompt asked the model to judge the *visual story* and to "evaluate the complete candidate", so it scored the whole raw 70-second window, even though the editing stage trims it to the best 15–45 seconds anyway, and it undervalued spoken payoffs.
**Fix and verification.** A revised prompt scores the best stretch and counts punchlines, roasts and reactions as payoffs, while keeping explicit rejections for inside jokes, banter and routine gameplay. It was A/B-tested on hand-labelled clips with identical inputs and switched on only because weak clips stayed at 0 passing (see [EVALUATION.md](EVALUATION.md)).

### 5. Muted audio making the speech model invent words
**Problem.** Some transcripts were nothing but "Thank you." at 0 s, 30 s and 60 s. The timestamps matched the speech model's 30-second processing chunks, and those sections of the VOD had been muted by the platform for copyrighted music. On silence the model fills in stock phrases, so the judge and the captions were working from invented text.
**Fix.** Measure the clip's loudness. Below −50 dB average it's marked *muted* and the judge is told the clip has no sound. Separately, stock filler lines that stretch over 8 seconds or more are dropped, since real speech never does that.

### 6. A crop that kept zooming in and out
**Problem.** Fight clips looked like the camera was shaking.
**Causes.** (a) When framing two people, their gap changed every frame and the zoom re-targeted constantly. (b) Smoothing constants were per frame, so 60 fps streams moved twice as fast as 30 fps ones. (c) When the tracker lost people for about 1.5 s it zoomed out to the full frame, then back in two seconds later. (d) A hard threshold switched between a full-height crop and a letterboxed view, so a width hovering near it flickered between the two.
**Fix.** The renderer plans the whole clip offline: a rolling median plus a 12 % dead-band on zoom, averaging plus a 6 % dead-zone on position, all in seconds rather than frames. A shot that would zoom out more than three times a minute stays wide.

### 7. Gameplay clips cropped onto the streamer's face
**Problem.** For a streamer with a face cam over gameplay, the tracker correctly found the person and cropped to them, so a plane crash happening in the game was entirely off-screen.
**Fix.** If the tracked subject is small (under 50 % of frame height) and barely moves all clip, it's a face cam. Those clips use a stacked layout: gameplay centre on top, the face cam enlarged below. Moving IRL subjects still get the normal crop.

### 8. Reliability: one bad reply shouldn't cost a 7-hour job
- A single malformed model reply crashed the judge and marked a whole VOD as failed. Now the JSON is pulled out of surrounding text, the request is retried once, and if it still fails only that one candidate is rejected.
- The disk cleanup deleted candidate videos the judge had *failed* on, as if they were rejected. Now only clips with a saved verdict are removed.
- Running out of API credits mid-run now stops the batch cleanly instead of producing half-finished clips.
- A full-audio download interrupted mid-way later failed transcription. The download is now validated, and skipped entirely when a transcript already exists.

### 9. "Corrupt" files that weren't corrupt
**Problem.** Some finished clips wouldn't play in browsers.
**Cause.** Encoders that didn't specify a pixel format inherited 4:4:4 (High 4:4:4 Predictive profile), which browsers and phones generally don't decode.
**Fix.** Every encoder now forces `yuv420p`, and existing affected clips were re-encoded.

### 10. Content safety in generated titles
**Problem.** A generated title mocked a person's physical trait. The prompt rule against it was ignored, because no single word was offensive; the combination was.
**Fix.** A code-level filter checks for a mocking verb near a physical or appearance word, plus explicit insults and slurs, and sends the title back through the model's rewrite loop. Checked against all previously generated titles: it blocked only the one problem title.
