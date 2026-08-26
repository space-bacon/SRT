---
title: A map of what a 27B understood
emoji: 🗺️
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
short_description: Click anywhere in a 27B's understanding and hear it
---

# A map of what a 27B understood

A **Qwen3.8-27B** read 123,287 photographs into a 1024-dimensional room, months
ago, on a datacenter GPU. It is not running here and never will again. What
survives is about **a kilobyte per photograph**.

This is that room, flattened to a page. Every dot is a photograph. **Click
anywhere** and a frozen **Qwen3-0.6B**, which has no vision path and has never
seen a photograph, says what is at that spot.

Every region name on the map was written the same way. We did not label the
continents. We handed the reader each cluster's centre and printed what it said.

## What is on it

The layout has structure you can navigate by. Animals to the west, transport to
the southwest, sport and open air along the north, domestic interiors down the
southeast. Nothing arranged that. It is where the photographs fell.

```
A couple of giraffes standing in a field
A herd of horses standing in a field
A man riding skis down a snow covered slope
A red and white bus is parked on the street
A room with a bed and a chair
A bathroom with a toilet, sink, and a mirror
```

Those are region names, not captions we chose.

## What you can do

**Click.** Anywhere, including the empty water between continents. The reader
takes the mean of the ~48 nearest photographs and describes it. Points between
regions are scenes no photograph shows, and it describes them anyway.

**Fly.** Type a sentence and a marker travels to where it belongs. *"a dog
asleep in the afternoon sun"* lands in the dog country and the eight nearest
photographs are all sleeping dogs. *"a man riding skis down a snowy slope"*
arrives at (0.10, 0.90) against a ski-region centre of (0.08, 0.86).

**Cross.** Two sentences, and it walks the ground between them, reading aloud as
it goes. The middle of the walk is somewhere nothing lives:

```
0.00  A dog sleeping in a bed in the sun.
0.25  A dog sleeping in a bed in the dark. A light is on. A person is walking by.
0.50  A dog sleeping in a car while a red light is on.
0.75  A red bus crossing a bridge in the rain.
1.00  A red bus crossing a bridge in the rain.
```

## Why this is not an ordinary embedding demo

Projections of embedding spaces are old, and cached image embeddings are
ordinary. Three things here are not.

**The regions named themselves.** Every UMAP scatterplot you have seen was
labelled by a human reading the cluster, or by the dataset's own metadata. These
names came out of a 0.6B model that was handed a centroid and asked what it was.

**The reader is a stranger to the writer.** It is not a decoder trained jointly
with its encoder. It is a separate, 45× smaller model reading a foreign
network's interior, and the controls show a shared representation is not what
carries it: hand the reader another photograph's point and it scores at chance.

**Arbitrary points speak, not just stored ones.** That is what makes this a map
rather than an index. You can stand where nothing is and still get an answer.

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

The map itself is UMAP (`n_neighbors=25`, `min_dist=0.12`, cosine) over 40,000
of the photographs, with 24 k-means regions. Two dimensions cannot carry 1,024,
so what you are looking at is neighbourhood structure. The coordinates mean
nothing on their own.

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
