# Aesthetic Principles for Impressive Main Figures

This document is about **visual design**, not analysis or template selection. Read it when:

- a figure is scientifically correct and structurally sound but still feels "ordinary"
- you want a figure to stop a reviewer mid-scroll
- you are targeting a top-tier venue (Nature Methods, Nature Biotechnology, Cell, etc.)

## The One Rule

A figure looks impressive when every pixel serves either **the message** or **the hierarchy**.
If a visual element does neither, remove it. Decoration without purpose is the fastest way to
make a figure look amateur.

---

## Part 1: Visual Hierarchy

### 1.1 Every panel needs exactly one focal point

The viewer's eye should land on one thing first. If everything competes, nothing wins.

Techniques to create focal points:

- **Saturation contrast**: foreground in full color, background in grey or low-alpha.
- **Size contrast**: the hero element is physically larger than supporting elements.
- **Isolation**: whitespace around the focal point separates it from noise.
- **Color singularity**: one accent color in an otherwise muted panel draws immediate attention.

Bad example: all cells on a UMAP at full saturation with 15 colors.
Good example: grey background cells, two saturated fate groups (red + blue), everything else quiet.

### 1.2 Three-layer visual stack

The best panels read as three layers of depth:

1. **Background** — context, orientation, quiet grey. The viewer barely notices it consciously.
2. **Midground** — supporting structure: axes, grid lines, reference lines, contours.
3. **Foreground** — the actual message: highlighted cells, trajectories, arrows, labels.

Control these layers with alpha, size, and color saturation:

```python
# Background: quiet
ax.scatter(bg_x, bg_y, c="#d0d0d0", s=1, alpha=0.3, edgecolors="none")
# Midground: structural
ax.contour(xx, yy, zz, levels=2, colors=["#1f77b4"], linewidths=0.8, alpha=0.5)
# Foreground: the message
ax.scatter(fg_x, fg_y, c="#1f77b4", s=6, alpha=0.8, edgecolors="none", zorder=5)
```

### 1.3 The squint test

Squint at your figure (or view it at 25% zoom). If you can still tell what each panel is about,
the hierarchy is working. If it looks like uniform noise, the hierarchy has failed.

---

## Part 2: Color

### 2.1 Less color is more impressive

Top-tier figures typically use **2-3 accent colors** plus grey, not the full matplotlib default
cycle. Restraint signals confidence.

A good palette for a 6-panel figure:

- One warm accent (red/orange) for group A
- One cool accent (blue/teal) for group B
- Grey for background/context
- Black for ink/text/frames
- Optional: one tertiary color (green/purple) for a third group

### 2.2 Color must mean something

Every non-grey color should encode a specific data variable or biological identity. If a color
is decorative, replace it with grey. Reviewers notice meaningless color faster than you think.

### 2.3 Consistent semantics across the entire figure

If BM is red in Panel A, it must be red everywhere. This sounds obvious but breaks constantly
during iterative figure development. Define colors as constants and import them:

```python
PALETTE = dict(BM="#d62728", AL="#1f77b4", FP="#2ca02c", BG="#d0d0d0", INK="#222222")
```

### 2.4 Colormaps: sequential for magnitudes, diverging for signed values

- **Sequential** (e.g., `viridis`, `magma`, `YlOrRd`): for probability, expression, intensity.
- **Diverging** (e.g., `RdBu_r`, `coolwarm`): for z-scores, fold changes, signed differences.
  Center at zero.
- Avoid `jet`, `rainbow`, `hsv`. They are perceptually non-uniform and signal 2005-era
  bioinformatics.

### 2.5 Quantile clipping instead of raw min/max

For scalar overlays on scatter plots, clip at [2nd, 98th] percentile before mapping to color.
This prevents one outlier from washing out the entire panel.

```python
vmin, vmax = np.percentile(values, [2, 98])
ax.scatter(x, y, c=values, vmin=vmin, vmax=vmax, cmap="magma")
```

---

## Part 3: Typography as Design

### 3.1 Font hierarchy is non-negotiable

- Panel letter: 11-12pt, bold, uppercase (A, B, C)
- Panel title: 9-10pt, bold
- Axis labels: 8-9pt, regular
- Tick labels: 7-8pt, regular
- Annotations/legends: 6-7pt

Never mix more than 2 font sizes in the same visual layer. Size variation signals hierarchy;
random variation signals chaos.

### 3.2 Direct labeling beats legends

Instead of a color legend with 6 entries, label the groups directly on the figure:

```python
for ct, (mx, my) in median_positions.items():
    ax.text(mx, my, ct, fontsize=7, fontweight="bold", ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
```

This is faster to read, removes legend clutter, and makes the panel feel more curated.

### 3.3 Italic for gene names, regular for everything else

Biological convention: gene names are always italicized (*Hes5*, *Sox9*). Using italic
correctly signals domain fluency.

```python
ax.set_yticklabels(gene_names, fontstyle="italic", fontsize=7)
```

---

## Part 4: Composition

### 4.1 Asymmetry > equal grids

Equal-sized panel grids feel like a homework assignment. Strong figures have **one larger hero
element** and **smaller supporting elements**:

```
| ████ hero ████ | small |
| ████ hero ████ | small |
| med  |  med  |  med   |
```

Allocate 40-55% of page height to the hero zone (top row).

### 4.2 Reading order: left-to-right, top-to-bottom

Panel A should be top-left. The narrative should flow naturally:
A (context) → B (model output) → C (validation) → D (dynamics) → E/F (mechanism).

Do not scatter related panels across non-adjacent positions.

### 4.3 Grammar changes create rhythm

A figure with 6 UMAP panels feels monotonous even if each shows different data. Vary the
visual grammar across rows:

- Row 1: UMAP scatter panels (geometry)
- Row 2: time-sliced trajectory panels (dynamics)
- Row 3: heatmap + circuit diagram (mechanism)

Each row change should feel like a deliberate shift in perspective, not a random mix.

### 4.4 Whitespace is structure, not waste

Generous gutters between panels help the eye group related items. Typical values:

- Within a panel group: 0.02-0.03 figure fraction
- Between panel groups (row breaks): 0.05-0.08 figure fraction
- Outer margins: 0.02-0.04 figure fraction

---

## Part 5: Techniques That Add Polish

### 5.1 KDE contours on scatter plots

Adding 1-2 levels of KDE contours to a scatter cluster instantly makes it look more
sophisticated. Use the same hue as the scatter points but at reduced alpha.

```python
from scipy.stats import gaussian_kde
kde = gaussian_kde(pts.T, bw_method=0.3)
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
ax.contour(xx, yy, zz, levels=2, colors=[color], linewidths=0.8, alpha=0.6)
```

### 5.2 Text halos for on-figure labels

When text overlaps dense scatter plots, add a white halo:

```python
from matplotlib.patheffects import withStroke
ax.text(x, y, label, fontsize=7, fontweight="bold",
        path_effects=[withStroke(linewidth=3, foreground="white")])
```

### 5.3 Soft edges on colorbars

Make colorbars feel integrated rather than pasted on:

```python
cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=15)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=6, length=2)
```

### 5.4 Consistent point sizes across panels

If Panel A uses `s=2` for background cells and Panel D also shows background cells, use `s=2`
there too. Inconsistent point sizes feel sloppy even if no one consciously notices.

### 5.5 `edgecolors="none"` everywhere

Remove point edges on scatter plots. The default thin black outline around each point makes
dense scatter plots look muddy. Always use `edgecolors="none"`.

### 5.6 Arrows that look intentional

For annotation arrows (e.g., pointing to a feature on UMAP):

```python
ax.annotate("", xy=(target_x, target_y), xytext=(start_x, start_y),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5,
                            connectionstyle="arc3,rad=0.1"))
```

Use straight arrows (`rad=0`) for circuit diagrams, slight curves (`rad=0.1-0.2`) for
annotation callouts. Never use the default matplotlib arrow style—it looks like a PowerPoint
autoshape.

### 5.7 Interpolated heatmaps

For trajectory heatmaps, `interpolation="bilinear"` makes the pseudo-time axis look smoother
and more continuous:

```python
ax.imshow(z, aspect="auto", cmap="RdBu_r", interpolation="bilinear")
```

### 5.8 Clean axis removal for manifold panels

For UMAP/embedding panels, don't just turn off ticks—remove everything:

```python
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
# Or simply:
ax.axis("off")
```

This makes manifold panels feel like standalone objects rather than trapped inside a chart frame.

---

## Part 6: What Makes Figures Look "Cheap"

Avoid these patterns. They are the most common reasons a figure fails to impress even when the
science is strong.

### 6.1 The "default matplotlib" look

- Grey figure background (use `facecolor="white"` explicitly)
- Blue default color on everything
- Default tick marks on embedding panels
- Visible top/right spines
- Matplotlib default figure title font

Fix: define `rcParams` at the top of every script or use `style_defaults.py`.

### 6.2 Rainbow everything

Using `tab20` or `Set3` on a UMAP with 15 cell types creates a confusing rainbow. Group similar
types into broader categories and use fewer, more deliberate colors. For >8 categories, consider
direct labeling instead of color-based legends.

### 6.3 Over-annotated heatmaps

Every cell in the heatmap has a number written in it. This turns a figure into a table.
Let the color do the work; add numbers only for the 3-5 most important cells.

### 6.4 Legends that dominate the panel

A legend taking up 30% of the panel area suggests the encoding is too complex. Simplify the
encoding, use direct labels, or move the legend to a shared location outside the panel.

### 6.5 Inconsistent panel spacing

Panels with different amounts of dead space around them look randomly assembled. Use
`fig.add_axes([x, y, w, h])` for precise control, or `gridspec` with consistent ratios.

### 6.6 Giant panel letters

Panel letters should be visible but not screaming. 14pt+ bold letters competing with the
figure content look like a textbook, not a journal article. Stick to 11-12pt.

### 6.7 Cluttered multi-panel figures with no breathing room

If you have 6 panels, you do not need to fill every pixel. Strategic whitespace makes the
figure feel composed. Think "magazine layout," not "surveillance dashboard."

---

## Part 7: The "Nature Methods" Checklist

Before declaring a main figure ready for a top-tier submission, check:

- [ ] Can I identify the hero panel in < 1 second?
- [ ] Does each panel have exactly one focal point?
- [ ] Are there ≤ 3 accent colors (excluding grey and black)?
- [ ] Is every color semantically consistent across all panels?
- [ ] Does the figure use ≥ 2 different visual grammars (e.g., scatter + heatmap + network)?
- [ ] Is the top row visually heavier than the bottom rows?
- [ ] Are all gene names in italic?
- [ ] Have I removed all unnecessary axis frames, ticks, and spines?
- [ ] Is every scatter plot using `edgecolors="none"`?
- [ ] Are colormaps perceptually uniform (no `jet` or `rainbow`)?
- [ ] Does the figure survive the squint test (identifiable at 25% zoom)?
- [ ] Is the PDF fully vector (no blurry elements at 400% zoom)?
- [ ] Would I be proud to present this figure at a conference?

---

## Part 8: Useful Matplotlib rcParams Presets

Place this at the top of every plotting script for a consistent baseline:

```python
import matplotlib as mpl

mpl.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "savefig.facecolor":   "white",
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           8,
    "axes.titlesize":      9,
    "axes.titleweight":    "bold",
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
})
```

---

## Summary

The distance between "correct figure" and "impressive figure" is usually not about adding more.
It is about:

1. **Removing** visual noise (default styles, unnecessary frames, redundant legends)
2. **Contrasting** foreground and background (saturation, alpha, size)
3. **Varying** grammar across the page (don't repeat the same plot type)
4. **Restraining** color (2-3 accents + grey is almost always enough)
5. **Polishing** details (halos, KDE contours, italic genes, clean colorbars)

The best figures feel like they were **designed**, not generated. That feeling comes from
intentionality at every level—from page skeleton to individual point alpha.
