import asyncio
from pathlib import Path

from open_rubric.rubric import Rubric
from open_rubric.viz.visualize import RubricVisualizer, visualize_rubric_from_path, VIZ_OUTPUT_DIR

rubric_path = "assets/example_rubrics/test_rubric.yaml"


def run() -> None:
    rubric = Rubric.from_yaml(rubric_path)
    
    # Create visualizer
    visualizer = RubricVisualizer(rubric)
    
    # Create output directory
    output_dir = VIZ_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    
    # Generate graphical visualizations
    visualizer.plot_networkx_graph(str(output_dir / "dependency_graph.png"))
    visualizer.create_graphviz_diagram(str(output_dir / "graphviz_diagram"))
    visualizer.create_plotly_interactive_graph(str(output_dir / "interactive_graph.html"))
    breakpoint()



if __name__ == "__main__":
    run()