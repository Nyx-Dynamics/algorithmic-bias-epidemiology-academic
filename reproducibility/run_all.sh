#!/bin/bash
# Run all analyses for Algorithmic Bias Epidemiology Academic Repository

set -e

echo "=============================================="
echo "Algorithmic Bias Epidemiology - Full Analysis"
echo "=============================================="

cd "$(dirname "$0")/.."

# Create output directories
mkdir -p data/simulation_results
mkdir -p manuscript/figures

# Run life success theorem analysis
echo ""
echo "Running Life Success Prevention Theorem..."
python analysis/life_success_theorem.py

# Run barrier visualization
echo ""
echo "Running Barrier Removal Analysis..."
python analysis/barrier_visualization.py

# Run population analysis
echo ""
echo "Running Population Attributable Fraction Analysis..."
python analysis/population_analysis.py

# Move outputs to proper directories
mv *.png manuscript/figures/ 2>/dev/null || true
mv *.csv data/simulation_results/ 2>/dev/null || true
mv *.json data/simulation_results/ 2>/dev/null || true

echo ""
echo "=============================================="
echo "Analysis Complete"
echo "=============================================="
echo "Figures saved to: manuscript/figures/"
echo "Data saved to: data/simulation_results/"
