# Panel Finish Checklist

Use this checklist after a panel is scientifically correct but before calling it ready.

The checklist exists because many figure problems are not failures of analysis or grammar.
They are finishing failures.

## 1. Typography

- Are panel letters in one consistent offset system?
- Are titles using one consistent hierarchy?
- Are axis labels, tick labels, and legend text readable at export scale?
- Is any font smaller than necessary just to force content to fit?
- Is bold used only for true hierarchy, not randomly?

## 2. Axes and Frame

- Does this panel follow the axis policy for its class?
- Are unnecessary ticks removed?
- Are unnecessary spines removed?
- Are reference lines present only when analytically meaningful?
- If the panel is manifold/spatial/latent, is equal aspect preserved?

## 3. Layout

- Does the plot box fill its allocated space without looking stretched?
- Is there too much dead white space?
- Is the panel too crowded near one edge?
- Are titles, labels, colorbars, and legends competing in the same gutter?
- Do sibling panels align along meaningful outer edges?

## 4. Legends and Labels

- Is the legend actually needed?
- If needed, is it compact and placed outside the data body when possible?
- Are on-data labels helping, or just cluttering the figure?
- Are long labels wrapped or abbreviated intentionally rather than colliding?

## 5. Visual Weight

- Does the panel look too thin?
- Does it rely on a single weak line when a band, box, or distribution would read better?
- Are the background and foreground clearly separated?
- Is one element visually shouting without being the main message?

## 6. Color

- Is the page background white unless explicitly requested otherwise?
- Is the palette coherent with the surrounding figure?
- Are accents used for data meaning, not decoration?
- Is the non-focus layer quiet enough?
- Are diverging palettes centered only when zero is meaningful?

## 7. Export Safety

- Are any labels clipped at the page edge?
- Are any labels clipped by neighboring panels or colorbars?
- Will the panel survive modest downscaling in a supplement or main-figure layout?
- Does the PNG/PDF look the same in overall balance?

## 8. Red Flags

If any answer below is yes, the panel is not finished:

- Does it read like a spreadsheet?
- Does it require a caption sentence to decode the encoding?
- Does the title rescue a weak panel rather than summarize a strong one?
- Is the panel only acceptable because the user is tired of revising it?

## Completion Standard

A panel is ready when:

- the grammar is correct
- the axis policy is correct
- the typography is stable
- the layout is clean
- the panel remains readable after export

This checklist is deliberately stricter than “looks okay on screen”.
