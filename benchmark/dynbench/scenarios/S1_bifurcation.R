#!/usr/bin/env Rscript
# =============================================================================
# DynBench Scenario S1: Simple Bifurcation (★★☆)
# Biological story: Progenitor → Fate A vs Fate B (monocyte/neutrophil-like)
# GRN: mutual inhibition toggle switch between two TFs
# =============================================================================

suppressPackageStartupMessages({
  library(scMultiSim)
  library(jsonlite)
  library(ape)
})

# ─── Output directory ────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else file.path(
  dirname(dirname(dirname(sys.frame(1)$ofile))),
  "ground_truth", "S1"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat("S1 output directory:", out_dir, "\n")

# ─── Simulation configuration ────────────────────────────────────────────────
set.seed(42)

# Bifurcation tree: Root → Fate_A + Fate_B
tree <- ape::read.tree(text = "((A:1,B:1):1);")

# GRN: toggle switch (mutual inhibition)
#   Gene_1 self-activates, inhibits Gene_2  → drives Fate A
#   Gene_2 self-activates, inhibits Gene_1  → drives Fate B
grn <- data.frame(
  target    = c(1, 2, 1, 2),
  regulator = c(1, 2, 2, 1),
  effect    = c(3.0, 3.0, -2.5, -2.5)
)

sim_options <- list(
  rand.seed = 42,
  GRN = grn,
  num.genes = 200,
  num.cells = 3000,
  num.cifs = 30,
  diff.cif.fraction = 0.80,
  tree = tree,
  do.velocity = TRUE,
  intrinsic.noise = 0.5,
  cif.sigma = 0.6,
  threads = 1
)

# ─── Run simulation ──────────────────────────────────────────────────────────
cat("Running scMultiSim simulation...\n")
results <- sim_true_counts(sim_options)
cat("Simulation complete.\n")

# ─── Extract results ─────────────────────────────────────────────────────────
meta <- as.data.frame(results$cell_meta)
expr <- as.data.frame(t(results$counts))

# Get pseudotime
if (!("pseudotime" %in% colnames(meta)) && !is.null(results$cell_time)) {
  meta$pseudotime <- as.numeric(results$cell_time)
} else if (!("pseudotime" %in% colnames(meta)) && "step" %in% colnames(meta)) {
  meta$pseudotime <- as.numeric(meta$step)
} else if (!("pseudotime" %in% colnames(meta))) {
  meta$pseudotime <- seq_len(nrow(meta))
}

# Get population/fate labels
if ("pop" %in% colnames(meta)) {
  meta$fate_true <- as.character(meta$pop)
} else {
  meta$fate_true <- "unknown"
}

# Gene names
colnames(expr) <- paste0("Gene_", seq_len(ncol(expr)))

# ─── Save raw outputs ────────────────────────────────────────────────────────
write.csv(expr, file.path(out_dir, "counts_cells_x_genes.csv"), row.names = TRUE)
write.csv(meta, file.path(out_dir, "full_metadata.csv"), row.names = TRUE)
saveRDS(results, file.path(out_dir, "scmultisim_results.rds"))

# Save velocity ground truth
if (!is.null(results$velocity)) {
  vel <- as.data.frame(t(results$velocity))
  colnames(vel) <- paste0("Gene_", seq_len(ncol(vel)))
  write.csv(vel, file.path(out_dir, "velocity_ground_truth.csv"), row.names = TRUE)
  cat("Velocity ground truth saved.\n")
}

# Save GRN design
grn_export <- list(
  edges = lapply(seq_len(nrow(grn)), function(i) {
    list(
      from = paste0("Gene_", grn$regulator[i]),
      to = paste0("Gene_", grn$target[i]),
      effect = grn$effect[i],
      direction = ifelse(grn$effect[i] > 0, "activate", "inhibit")
    )
  }),
  regulators = unique(paste0("Gene_", c(grn$regulator, grn$target)))
)
write_json(grn_export, file.path(out_dir, "grn_input.json"),
           auto_unbox = TRUE, pretty = TRUE)

# ─── Summary ─────────────────────────────────────────────────────────────────
cat("\n=== S1 Simulation Summary ===\n")
cat("Cells:", nrow(meta), "\n")
cat("Genes:", ncol(expr), "\n")
cat("Pseudotime range:", range(meta$pseudotime), "\n")
cat("Fates:", paste(unique(meta$fate_true), collapse = ", "), "\n")
fate_counts <- table(meta$fate_true)
cat("Fate counts:\n")
print(fate_counts)
cat("Velocity available:", !is.null(results$velocity), "\n")
cat("\nAll outputs saved to:", out_dir, "\n")
