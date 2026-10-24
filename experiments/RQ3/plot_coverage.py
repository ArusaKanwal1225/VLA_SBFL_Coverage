#!/usr/bin/env python3
"""Build the RQ3 training-coverage versus ROC-AUC scatter visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TECHNIQUES = ("dennis", "harmonic_mean", "rogot2", "tarwid")


def aggregate_plot_data(metrics_path: Path) -> tuple[list[dict], list[dict]]:
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["formula_id"].isin(TECHNIQUES)].copy()
    configuration = metrics.groupby([
        "configuration_id", "vla", "condition", "budget", "training_size",
        "coverage_tier", "formula_id", "technique",
    ], as_index=False).agg(
        roc_auc=("roc_auc", "mean"),
        coverage=("calibration_unique_coverage", "mean"),
    )
    block = configuration.groupby([
        "vla", "condition", "training_size", "coverage_tier", "formula_id",
        "technique",
    ], as_index=False).agg(
        roc_auc=("roc_auc", "mean"),
        coverage=("coverage", "mean"),
    )
    across_sizes = block.groupby([
        "vla", "training_size", "formula_id", "technique",
    ], as_index=False).agg(
        roc_auc=("roc_auc", "mean"),
        coverage=("coverage", "mean"),
    )
    fixed_size = block.groupby([
        "training_size", "coverage_tier", "formula_id", "technique",
    ], as_index=False).agg(
        roc_auc=("roc_auc", "mean"),
        coverage=("coverage", "mean"),
    )
    for frame in (across_sizes, fixed_size):
        frame["roc_auc"] = frame["roc_auc"].round(5)
        frame["coverage"] = frame["coverage"].round(5)
    return (
        across_sizes.to_dict(orient="records"),
        fixed_size.to_dict(orient="records"),
    )


def build_fragment(across_sizes: list[dict], fixed_size: list[dict]) -> str:
    template = r'''<div id="rq3-coverage-scatter" class="rq3-coverage-scatter">
  <div class="rq3-legend viz-row" aria-label="Technique legend"></div>
  <svg class="rq3-chart rq3-overview" viewBox="0 0 736 548" role="img" aria-label="Training coverage versus ROC-AUC for four techniques, faceted by VLA and connected across training sizes"></svg>
  <svg class="rq3-chart rq3-controlled" viewBox="0 0 736 410" role="img" aria-label="Training coverage versus ROC-AUC at each fixed training size, comparing low, medium, and high coverage subsets"></svg>
  <div class="tooltip rq3-tooltip" aria-hidden="true"></div>
  <p class="sr-only">The first plot shows a strong positive association when training size increases. The second plot holds training size fixed and shows small, inconsistent changes between coverage tiers.</p>
</div>
<style>
#rq3-coverage-scatter { position: relative; width: 100%; color: var(--foreground); }
#rq3-coverage-scatter .rq3-legend { justify-content: center; gap: 14px; margin-bottom: 4px; }
#rq3-coverage-scatter .rq3-legend-item { display: inline-flex; align-items: center; gap: 6px; color: var(--foreground); }
#rq3-coverage-scatter .rq3-swatch { width: 18px; height: 3px; display: inline-block; background: var(--series); }
#rq3-coverage-scatter .rq3-chart { display: block; width: 100%; height: auto; overflow: visible; }
#rq3-coverage-scatter .rq3-axis { stroke: var(--border); stroke-width: 1; }
#rq3-coverage-scatter .rq3-grid { stroke: var(--border); stroke-width: 1; opacity: .55; }
#rq3-coverage-scatter .rq3-label { fill: var(--muted-foreground); font-size: 11px; font-weight: 400; }
#rq3-coverage-scatter .rq3-facet { fill: var(--foreground); font-size: 13px; font-weight: 500; }
#rq3-coverage-scatter .rq3-section { fill: var(--foreground); font-size: 13px; font-weight: 500; }
#rq3-coverage-scatter .rq3-line { fill: none; stroke-width: 2; opacity: .88; }
#rq3-coverage-scatter .rq3-point { stroke: var(--background); stroke-width: 1.5; cursor: default; }
#rq3-coverage-scatter .rq3-size-label { fill: var(--muted-foreground); font-size: 11px; }
#rq3-coverage-scatter .rq3-tooltip { position: absolute; display: none; pointer-events: none; max-width: 210px; }
</style>
<script>
(() => {
  const root = document.getElementById('rq3-coverage-scatter');
  const overviewData = __OVERVIEW_DATA__;
  const fixedData = __FIXED_DATA__;
  const ns = 'http://www.w3.org/2000/svg';
  const techniques = [
    { id: 'dennis', label: 'Dennis', color: 'var(--viz-series-1)' },
    { id: 'harmonic_mean', label: 'Harmonic Mean', color: 'var(--viz-series-2)' },
    { id: 'rogot2', label: 'Rogot2', color: 'var(--viz-series-3)' },
    { id: 'tarwid', label: 'Tarwid', color: 'var(--viz-series-4)' }
  ];
  const vlaOrder = ['EO1', 'Groot', 'Pi0', 'SpatialVLA'];
  const sizeOrder = [10, 20, 50, 100, 200];
  const tierOrder = ['low', 'medium', 'high'];
  const tierLabel = { low: 'Low', medium: 'Medium', high: 'High' };

  function node(name, attrs = {}, text = '') {
    const element = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    if (text) element.textContent = text;
    return element;
  }

  function scale(domainMin, domainMax, rangeMin, rangeMax) {
    return value => rangeMin + (value - domainMin) * (rangeMax - rangeMin) / (domainMax - domainMin);
  }

  function linePath(points) {
    return points.map((point, index) => `${index ? 'L' : 'M'}${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(' ');
  }

  function showTooltip(mark, datum) {
    const tooltip = root.querySelector('.rq3-tooltip');
    tooltip.innerHTML = `<strong>${datum.technique}</strong><br>${datum.vla ? datum.vla + '<br>' : ''}Training: ${datum.training_size} trajectories${datum.coverage_tier ? '<br>Tier: ' + tierLabel[datum.coverage_tier] : ''}<br>Coverage: ${(100 * datum.coverage).toFixed(1)}%<br>ROC-AUC: ${datum.roc_auc.toFixed(3)}`;
    tooltip.style.display = 'block';
    tooltip.setAttribute('aria-hidden', 'false');
    const markBox = mark.getBoundingClientRect();
    const rootBox = root.getBoundingClientRect();
    const tooltipBox = tooltip.getBoundingClientRect();
    const left = Math.max(4, Math.min(markBox.left - rootBox.left + markBox.width / 2 - tooltipBox.width / 2, rootBox.width - tooltipBox.width - 4));
    const top = Math.max(4, markBox.top - rootBox.top - tooltipBox.height - 8);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function hideTooltip() {
    const tooltip = root.querySelector('.rq3-tooltip');
    tooltip.style.display = 'none';
    tooltip.setAttribute('aria-hidden', 'true');
  }

  function addPoint(svg, datum, x, y, color, shape = 'circle') {
    let mark;
    if (shape === 'square') {
      mark = node('rect', { x: x - 4, y: y - 4, width: 8, height: 8, rx: 1 });
    } else if (shape === 'triangle') {
      mark = node('path', { d: `M${x},${y - 5} L${x + 5},${y + 4} L${x - 5},${y + 4} Z` });
    } else {
      mark = node('circle', { cx: x, cy: y, r: 4 });
    }
    mark.setAttribute('class', 'rq3-point');
    mark.setAttribute('fill', color);
    mark.addEventListener('mouseenter', () => showTooltip(mark, datum));
    mark.addEventListener('mouseleave', hideTooltip);
    svg.appendChild(mark);
  }

  function addAxes(svg, box, xDomain, yDomain, xTicks, yTicks, showXLabel, showYLabel) {
    const xScale = scale(xDomain[0], xDomain[1], box.x, box.x + box.w);
    const yScale = scale(yDomain[0], yDomain[1], box.y + box.h, box.y);
    yTicks.forEach(value => {
      const y = yScale(value);
      svg.appendChild(node('line', { x1: box.x, y1: y, x2: box.x + box.w, y2: y, class: 'rq3-grid' }));
      svg.appendChild(node('text', { x: box.x - 7, y: y + 4, 'text-anchor': 'end', class: 'rq3-label' }, value.toFixed(2)));
    });
    xTicks.forEach(value => {
      const x = xScale(value);
      svg.appendChild(node('line', { x1: x, y1: box.y, x2: x, y2: box.y + box.h, class: 'rq3-grid' }));
      svg.appendChild(node('text', { x, y: box.y + box.h + 17, 'text-anchor': 'middle', class: 'rq3-label' }, `${Math.round(100 * value)}%`));
    });
    svg.appendChild(node('line', { x1: box.x, y1: box.y + box.h, x2: box.x + box.w, y2: box.y + box.h, class: 'rq3-axis' }));
    svg.appendChild(node('line', { x1: box.x, y1: box.y, x2: box.x, y2: box.y + box.h, class: 'rq3-axis' }));
    if (showXLabel) svg.appendChild(node('text', { x: box.x + box.w / 2, y: box.y + box.h + 36, 'text-anchor': 'middle', class: 'rq3-label' }, 'Training coverage'));
    if (showYLabel) {
      const label = node('text', { x: box.x - 42, y: box.y + box.h / 2, 'text-anchor': 'middle', class: 'rq3-label', transform: `rotate(-90 ${box.x - 42} ${box.y + box.h / 2})` }, 'ROC-AUC');
      svg.appendChild(label);
    }
    return { xScale, yScale };
  }

  const legend = root.querySelector('.rq3-legend');
  techniques.forEach(technique => {
    const item = document.createElement('span');
    item.className = 'rq3-legend-item text-small';
    item.style.setProperty('--series', technique.color);
    item.innerHTML = `<span class='rq3-swatch' aria-hidden='true'></span>${technique.label}`;
    legend.appendChild(item);
  });

  const overview = root.querySelector('.rq3-overview');
  overview.appendChild(node('text', { x: 368, y: 18, 'text-anchor': 'middle', class: 'rq3-section' }, 'Across training sizes · coverage tiers averaged'));
  vlaOrder.forEach((vla, facetIndex) => {
    const column = facetIndex % 2;
    const row = Math.floor(facetIndex / 2);
    const box = { x: 58 + column * 364, y: 56 + row * 248, w: 292, h: 174 };
    overview.appendChild(node('text', { x: box.x + box.w / 2, y: box.y - 13, 'text-anchor': 'middle', class: 'rq3-facet' }, vla));
    const axes = addAxes(overview, box, [0, 0.72], [0.49, 0.70], [0, 0.2, 0.4, 0.6], [0.50, 0.55, 0.60, 0.65, 0.70], row === 1, column === 0);
    techniques.forEach(technique => {
      const series = overviewData.filter(datum => datum.vla === vla && datum.formula_id === technique.id).sort((a, b) => a.training_size - b.training_size);
      const points = series.map(datum => [axes.xScale(datum.coverage), axes.yScale(datum.roc_auc)]);
      overview.appendChild(node('path', { d: linePath(points), class: 'rq3-line', stroke: technique.color }));
      series.forEach((datum, index) => {
        addPoint(overview, datum, points[index][0], points[index][1], technique.color);
        if (technique.id === 'dennis') {
          overview.appendChild(node('text', { x: points[index][0] + 6, y: points[index][1] - 6, class: 'rq3-size-label' }, String(datum.training_size)));
        }
      });
    });
  });

  const controlled = root.querySelector('.rq3-controlled');
  controlled.appendChild(node('text', { x: 368, y: 18, 'text-anchor': 'middle', class: 'rq3-section' }, 'Within fixed training sizes · low, medium, and high coverage'));
  controlled.appendChild(node('text', { x: 368, y: 38, 'text-anchor': 'middle', class: 'rq3-label' }, 'Point shape: low ○   medium □   high △'));
  sizeOrder.forEach((trainingSize, facetIndex) => {
    const column = facetIndex % 3;
    const row = Math.floor(facetIndex / 3);
    const box = { x: 50 + column * 238, y: 75 + row * 166, w: 188, h: 105 };
    const facetData = fixedData.filter(datum => datum.training_size === trainingSize);
    const coverages = facetData.map(datum => datum.coverage);
    const xMin = Math.max(0, Math.min(...coverages) - 0.008);
    const xMax = Math.max(...coverages) + 0.008;
    controlled.appendChild(node('text', { x: box.x + box.w / 2, y: box.y - 11, 'text-anchor': 'middle', class: 'rq3-facet' }, `${trainingSize} trajectories`));
    const axes = addAxes(controlled, box, [xMin, xMax], [0.53, 0.66], [xMin, xMax], [0.54, 0.60, 0.66], row === 1, column === 0);
    techniques.forEach(technique => {
      const series = facetData.filter(datum => datum.formula_id === technique.id).sort((a, b) => a.coverage - b.coverage);
      const points = series.map(datum => [axes.xScale(datum.coverage), axes.yScale(datum.roc_auc)]);
      controlled.appendChild(node('path', { d: linePath(points), class: 'rq3-line', stroke: technique.color }));
      series.forEach((datum, index) => {
        const shape = datum.coverage_tier === 'medium' ? 'square' : datum.coverage_tier === 'high' ? 'triangle' : 'circle';
        addPoint(controlled, datum, points[index][0], points[index][1], technique.color, shape);
      });
    });
  });
})();
</script>
'''
    return template.replace(
        "__OVERVIEW_DATA__", json.dumps(across_sizes, separators=(",", ":"))
    ).replace(
        "__FIXED_DATA__", json.dumps(fixed_size, separators=(",", ":"))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("experiments/RQ3/results/phase_a_coverage/aggregate/cell_metrics.csv.gz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    across_sizes, fixed_size = aggregate_plot_data(args.metrics)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_fragment(across_sizes, fixed_size), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
