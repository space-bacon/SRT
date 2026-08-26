---
title: Walk the space
emoji: 🚶
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - interpretability
  - cross-modal
  - representation-learning
  - embeddings
  - activation-verbalization
  - vision-language
short_description: Hear what is anywhere in a 27B's semantic space
---

# Walk the space

A **Qwen3.8-27B** read 123,287 photographs into a 1024-dimensional room, months
ago, on a datacenter GPU. It is not running here and never will again. What
survives is about **a kilobyte per photograph**.

A **2.1 MB** linear head puts your sentences into that same room. And a frozen
**Qwen3-0.6B**, which has no vision path and has never seen a photograph, says
what is at any point in it.

That last part is what turns the room into a place rather than a lookup table.
You can stand where a photograph is, where your own words are, or **halfway
between two photographs, where nothing is at all**, and hear what is there.

## What you can do

**Travel.** Give it two sentences and cross the ground between them. Every point
in the middle is a scene that does not exist, and the reader describes it
anyway:

```
0.00  A dog sleeping in a bed in the sun.
0.25  A dog sleeping in a bed in the dark. A light is on. A person is walking by.
0.50  A dog sleeping in a car while a red light is on.
0.75  A red bus crossing a bridge in the rain.
1.00  A red bus crossing a bridge in the rain.
```

**Arithmetic.** An axis is the difference between two groups of words. Add it to
where you are standing, and listen to where you end up:

```
−1.0  toward vehicles   A car is going down a street with a train coming.
 0.0  your sentence     A group of people walking down a street. A car is coming.
+1.0  toward animals    A large animal is running down a street. A dog is running.
```

**Round trip.** Type a sentence with no destination. It is encoded to 1,024
numbers by one model and read back by another that never saw your text. *"A red
bus crossing a bridge in the rain"* comes back as *"A red bus crossing a bridge
in the rain."*

## Why this is not an ordinary embedding demo

Vector arithmetic is decades old and cached image embeddings are ordinary. Three
things here are not.

**The reader is a stranger to the writer.** It is not a decoder trained
jointly with its encoder. It is a separate, 45× smaller model reading a foreign
network's interior, and the controls show a shared representation is not what
carries it: hand the reader another photograph's point and it scores at chance.

**Arbitrary points speak, not just stored ones.** The midpoint between two
photographs is somewhere no photograph lives. It still says something coherent,
and something that belongs to neither end.

**The large model is a finished file.** Its understanding did not need to be
recomputed. It needed to be stored in a format something else could read.

Which is the actual claim: machine understanding of an image compresses to about
a kilobyte, and that kilobyte is enough for a much smaller, unrelated model to
say what is in it.

## How it works

```
your sentence ──► Qwen3-0.6B ──► 2.1 MB head ──┐
                                                ├──► a point in the room (1024-d)
photographs ──► Qwen3.8-27B (once, offline) ────┘
                                                        │
                                     36 MB adapter ──► Qwen3-0.6B ──► a sentence
```

The reader is a prefix network that turns one point into 16 soft tokens, which
a frozen Qwen3-0.6B speaks from. Only the prefix was trained; the backbone is
untouched, and it is the same weights that encode your query.

## The honest part

The reader is fitted to the region of the room where photographs live. Points
far outside it drift toward generic scenes rather than failing loudly, so a
sentence about something COCO never photographed will come back vaguer than you
expect. That is a real boundary and you can find it in a couple of minutes.

It also restates itself, because it was trained on captions that carried no
end-of-sequence token and never learned that a description ends. Exact repeats
are dropped for display; nothing is reordered or invented.

Measured on 5,000 held-out photographs, a sentence it writes retrieves the
photograph it was written from at **median rank 18 of 123,287**. The same
reader handed another photograph's point scores at **chance**. Full evaluation,
including three negative results, is on the
[model card](https://huggingface.co/RiverRider/srt-verbalizer-v1).

## Provenance

- Reader checkpoints: [`RiverRider/srt-verbalizer-v1`](https://huggingface.co/RiverRider/srt-verbalizer-v1)
- Head and gallery: [`RiverRider/srt-browser-head-118k`](https://huggingface.co/RiverRider/srt-browser-head-118k)
- Code, papers and every banked measurement: [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT)
- The same instrument, running entirely in your browser with no server:
  [0.6b-reads-27b](https://huggingface.co/spaces/RiverRider/0.6b-reads-27b)

Built by [Sunstone North](https://sunstonenorth.com), the engineering studio of
the SRT research programme.
