# Generic transition alluvial template
#
# Inspired by moslin zebrafish alluvial plots.
# Input data should already be in long form with one row per transition flow.
#
# Required columns:
# - time
# - state
# - alluvium
# - fraction
# - fill_group

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggalluvial)
})

plot_transition_alluvial <- function(df,
                                     x_col = "time",
                                     stratum_col = "state",
                                     alluvium_col = "alluvium",
                                     y_col = "fraction",
                                     fill_col = "fill_group",
                                     palette = NULL) {
  p <- ggplot(
    df,
    aes_string(
      x = x_col,
      stratum = stratum_col,
      alluvium = alluvium_col,
      y = y_col,
      fill = fill_col,
      label = stratum_col
    )
  ) +
    geom_flow(alpha = 0.85) +
    geom_stratum(size = 0) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_classic() +
    theme(
      legend.position = "none",
      panel.background = element_blank()
    )

  if (!is.null(palette)) {
    p <- p + scale_fill_manual(values = palette)
  }

  return(p)
}

