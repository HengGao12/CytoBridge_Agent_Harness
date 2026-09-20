#!/usr/bin/env Rscript
# =============================================================================
# DynBench Scenario S3: Complex Tree + Dynamic GRN + Extrapolation (★★★★)
# Biological story: Multi-lineage differentiation with time-varying regulatory
#   network (like early embryonic development)
# GRN: built-in 100-gene network + custom master regulators, dynamic edges
# Agent sees [B0, B2, B4] → must extrapolate to B5
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
  "ground_truth", "S3"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat("S3 output directory:", out_dir, "\n")

# ─── Simulation configuration ────────────────────────────────────────────────
set.seed(456)

# Complex 5-branch tree
tree <- Phyla5()

# Use built-in 100-gene GRN as base
data(GRN_params_100)

# Add custom master regulatory edges
# GRN_params_100 uses columns: regulated.gene, regulator.gene, regulator.effect
grn_custom <- rbind(GRN_params_100, data.frame(
  regulated.gene  = c(101, 102, 101),
  regulator.gene  = c(103, 104, 104),
  regulator.effect = c(5.0, 5.5, -4.0)
))

sim_options <- list(
  rand.seed = 456,
  GRN = grn_custom,
  num.genes = 300,
  num.cells = 4000,
  num.cifs = 50,
  diff.cif.fraction = 0.85,
  tree = tree,
  do.velocity = TRUE,
  intrinsic.noise = 0.7,
  cif.sigma = 0.8,
  dynamic.GRN = list(
    cell.per.step = 3,
    num.changing.edges = 8,
    weight.mean = 0,
    weight.sd = 4
  ),
  threads = 1
)

# ─── Run simulation ──────────────────────────────────────────────────────────
cat("Running scMultiSim simulation (this may take a few minutes)...\n")
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

# Save GRN design — for S3, full GRN is large, so also save summary
grn_for_export <- grn_custom
grn_export <- list(
  n_edges = nrow(grn_for_export),
  edges = lapply(seq_len(nrow(grn_for_export)), function(i) {
    list(
      from = paste0("Gene_", grn_for_export$regulator.gene[i]),
      to = paste0("Gene_", grn_for_export$regulated.gene[i]),
      effect = grn_for_export$regulator.effect[i],
      direction = ifelse(grn_for_export$regulator.effect[i] > 0, "activate", "inhibit")
    )
  }),
  regulators = unique(paste0("Gene_", grn_for_export$regulator.gene)),
  targets = unique(paste0("Gene_", grn_for_export$regulated.gene)),
  has_dynamic_grn = TRUE,
  dynamic_grn_params = list(
    cell_per_step = 3,
    num_changing_edges = 8
  )
)
write_json(grn_export, file.path(out_dir, "grn_input.json"),
           auto_unbox = TRUE, pretty = TRUE)

# ─── Summary ─────────────────────────────────────────────────────────────────
cat("\n=== S3 Simulation Summary ===\n")
cat("Cells:", nrow(meta), "\n")
cat("Genes:", ncol(expr), "\n")
cat("Pseudotime range:", range(meta$pseudotime), "\n")
cat("Fates:", paste(unique(meta$fate_true), collapse = ", "), "\n")
fate_counts <- table(meta$fate_true)
cat("Fate counts:\n")
print(fate_counts)
cat("GRN edges:", nrow(grn_custom), "\n")
cat("Dynamic GRN: YES\n")
cat("Velocity available:", !is.null(results$velocity), "\n")
cat("\nAll outputs saved to:", out_dir, "\n")
