import typing as t
from pathlib import Path

import graphviz
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

from open_rubric.rubric import Rubric
from open_rubric.utils.dag import topological_levels


class RubricVisualizer:
    """Comprehensive visualization toolkit for Open Rubric system."""

    def __init__(self, rubric: Rubric):
        self.rubric = rubric
        self.requirements = rubric.requirements.get_all_requirements()
        self.dependencies = rubric.requirements.dependencies
        self.execution_levels = topological_levels(self.dependencies)

    def _format_component_set(self, component_set: set[tuple[str, str]]) -> str:
        """Helper method to format component sets into readable strings.

        Args:
            component_set: Set of (name, type) tuples
            with_spaces: Whether to include spaces around parentheses

        Returns:
            Formatted string like "name1 (type1), name2 (type2)" or "name1(type1), name2(type2)"
        """
        return ", ".join([f"{name} ({type_})" for name, type_ in component_set])

    def create_networkx_graph(self) -> nx.DiGraph:
        """Create NetworkX graph of requirements with component references."""
        G = nx.DiGraph()

        # Add requirement nodes with component references
        for req in self.requirements:
            # Find which execution level this requirement belongs to
            execution_level = None
            for i, level in enumerate(self.execution_levels):
                if req.name in level:
                    execution_level = i + 1
                    break

            # Check completion status
            has_result = req._result is not None
            query_answered = req.query.been_answered
            score = None
            confidence = None

            if has_result:
                assert req._result is not None
                score = req._result.score
                confidence = req._result.confidence
            elif query_answered:
                score = req.query.score

            G.add_node(
                req.name,
                node_type="requirement",
                title=req.name,
                execution_level=execution_level,
                evaluator_name=req.evaluator.name,
                evaluator_type=req.evaluator.type,
                scoring_type=req.query.scoring_config.subtype,
                aggregator_name=req.aggregator.name,
                instruction=req.query.instruction,
                full_instruction=req.query.instruction,
                has_result=has_result,
                query_answered=query_answered,
                score=score,
                confidence=confidence,
                completion_status=(
                    "completed" if has_result else ("answered" if query_answered else "pending")
                ),
            )

        # Add edges for requirement dependencies only
        for req_name, deps in self.dependencies.items():
            if deps:
                for dep in deps:
                    G.add_edge(dep, req_name, edge_type="dependency")

        return G

    def get_component_summary(self) -> dict[str, set]:
        """Get summary of all components used in the rubric."""
        evaluators = set()
        scoring_configs = set()
        aggregators = set()

        for req in self.requirements:
            evaluators.add((req.evaluator.name, req.evaluator.type))
            scoring_configs.add((req.query.scoring_config.name, req.query.scoring_config.type))
            aggregators.add((req.aggregator.name, req.aggregator.subtype))

        return {
            "evaluators": evaluators,
            "scoring_configs": scoring_configs,
            "aggregators": aggregators,
        }

    def get_completion_summary(self) -> dict[str, int]:
        """Get summary of completion status across all requirements."""
        completed = sum(1 for req in self.requirements if req._result is not None)
        answered = sum(
            1 for req in self.requirements if req.query.been_answered and req._result is None
        )
        pending = len(self.requirements) - completed - answered

        return {
            "completed": completed,
            "answered": answered,
            "pending": pending,
            "total": len(self.requirements),
        }

    def plot_networkx_graph(
        self, save_path: t.Optional[str] = None, figsize: tuple[int, int] = (14, 10)
    ) -> None:
        """Plot clean NetworkX graph with requirements and component reference panel."""
        G = self.create_networkx_graph()
        components = self.get_component_summary()
        completion = self.get_completion_summary()

        fig, (ax_components, ax_graph) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 4]}
        )

        # Top panel: Component reference with completion status - all in vertical list
        ax_components.text(
            0.02,
            0.9,
            "COMPLETION STATUS:",
            fontweight="bold",
            fontsize=10,
            transform=ax_components.transAxes,
        )
        completion_text = f"✓ Completed: {completion['completed']}, ○ Answered: {completion['answered']}, - Pending: {completion['pending']} / {completion['total']}"
        ax_components.text(
            0.02, 0.8, completion_text, fontsize=9, transform=ax_components.transAxes
        )

        ax_components.text(
            0.02,
            0.65,
            "EVALUATORS:",
            fontweight="bold",
            fontsize=10,
            transform=ax_components.transAxes,
        )
        eval_text = self._format_component_set(components["evaluators"])
        ax_components.text(
            0.02, 0.55, eval_text, fontsize=9, transform=ax_components.transAxes, wrap=True
        )

        ax_components.text(
            0.02, 0.4, "SCORING:", fontweight="bold", fontsize=10, transform=ax_components.transAxes
        )
        scoring_text = self._format_component_set(components["scoring_configs"])
        ax_components.text(0.02, 0.3, scoring_text, fontsize=9, transform=ax_components.transAxes)

        ax_components.text(
            0.02,
            0.15,
            "AGGREGATORS:",
            fontweight="bold",
            fontsize=10,
            transform=ax_components.transAxes,
        )
        agg_text = self._format_component_set(components["aggregators"])
        ax_components.text(0.02, 0.05, agg_text, fontsize=9, transform=ax_components.transAxes)

        ax_components.set_xlim(0, 1)
        ax_components.set_ylim(0, 1)
        ax_components.axis("off")
        ax_components.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="lightgray", alpha=0.3))

        # Bottom panel: Requirements graph
        plt.sca(ax_graph)

        # Create hierarchical layout with more spacing for larger nodes
        pos = {}
        level_height = 3.0  # Increased spacing between levels
        level_width = 8.0  # Increased width for larger nodes

        for i, level in enumerate(self.execution_levels):
            y_pos = (len(self.execution_levels) - i - 1) * level_height
            n_nodes = len(level)

            for j, node in enumerate(level):
                if n_nodes == 1:
                    x_pos = 0.0
                else:
                    x_pos = (j - (n_nodes - 1) / 2) * (level_width / max(n_nodes - 1, 1))
                pos[node] = (x_pos, y_pos)

        # Color nodes by completion status instead of evaluator type
        completion_color_map = {
            "completed": "#90EE90",  # Light green
            "answered": "#FFE4B5",  # Light yellow/orange
            "pending": "#FFB6C1",  # Light pink
        }

        node_colors = [
            completion_color_map[G.nodes[node]["completion_status"]] for node in G.nodes()
        ]

        # Draw execution level backgrounds
        for i, _ in enumerate(self.execution_levels):
            y_pos = (len(self.execution_levels) - i - 1) * level_height
            rect = plt.Rectangle(
                (-level_width / 2 - 0.7, y_pos - 0.6),
                level_width + 1.4,
                1.2,
                facecolor="lightgray",
                alpha=0.2,
                edgecolor="gray",
            )
            ax_graph.add_patch(rect)
            ax_graph.text(
                -level_width / 2 - 0.5,
                y_pos,
                f"Level {i + 1}",
                rotation=90,
                va="center",
                ha="center",
                fontweight="bold",
            )

        # Draw graph with larger nodes
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            node_size=3500,  # Much larger nodes to contain text
            font_size=6,  # Smaller font to fit in larger nodes
            font_weight="bold",
            arrows=True,
            arrowsize=20,
            edge_color="darkblue",
            width=2,
            alpha=0.8,
            ax=ax_graph,
        )

        # Add requirement labels with component info and completion status
        for node in G.nodes():
            attrs = G.nodes[node]
            x, y = pos[node]

            # Main title above the node with completion status emoji
            completion_emoji = {"completed": "✓", "answered": "○", "pending": "-"}[
                attrs["completion_status"]
            ]
            title_text = f"{completion_emoji} {attrs['title']}"
            ax_graph.text(
                x, y + 0.4, title_text, ha="center", va="center", fontsize=8, fontweight="bold"
            )

            # Component info below the node in smaller text
            component_text = f"E:{attrs['evaluator_name'][:8]}\nS:{attrs['scoring_type'][:8]}\nA:{attrs['aggregator_name'][:8]}"

            # Add score if available
            if attrs["score"] is not None:
                score_str = f"{attrs['score']}"
                if attrs["confidence"] is not None:
                    score_str += f" (conf: {attrs['confidence']:.2f})"
                component_text += f"\nScore: {score_str}"

            ax_graph.text(
                x,
                y - 0.5,
                component_text,
                ha="center",
                va="center",
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
            )

        # Create legend for completion status
        legend_patches = [
            mpatches.Patch(
                color=completion_color_map["completed"], label="✓ Completed (has result)"
            ),
            mpatches.Patch(color=completion_color_map["answered"], label="○ Answered (query only)"),
            mpatches.Patch(color=completion_color_map["pending"], label="- Pending"),
        ]
        ax_graph.legend(
            handles=legend_patches,
            title="Completion Status",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

        ax_graph.set_title(
            "Requirements Dependencies with Completion Status\n(Organized by Execution Levels)",
            fontsize=14,
            fontweight="bold",
        )
        ax_graph.axis("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graph saved to {save_path}")

    def create_graphviz_diagram(self, save_path: t.Optional[str] = None) -> str:
        """Create clean Graphviz diagram with component reference."""
        dot = graphviz.Digraph(comment="Clean Rubric Structure")
        dot.attr(rankdir="TB", size="12,10!")
        dot.attr(
            "node", shape="box", style="rounded,filled", width="1.5", height="1.0"
        )  # Larger nodes

        components = self.get_component_summary()

        # Add component reference at top
        ref_label = "COMPONENTS REFERENCE\\n\\n"
        ref_label += "EVALUATORS: " + self._format_component_set(components["evaluators"]) + "\\n"
        ref_label += "SCORING: " + self._format_component_set(components["scoring_configs"]) + "\\n"
        ref_label += "AGGREGATORS: " + self._format_component_set(components["aggregators"])

        dot.node(
            "reference",
            ref_label,
            shape="note",
            fillcolor="lightgray",
            fontsize="8",
            width="4",
            height="1.5",
        )

        # Define colors for different evaluator types
        # evaluator_types = list(set(req.evaluator.type for req in self.requirements))
        evaluator_colors = {"llm": "#FFE5E5", "llm-ensemble": "#E5F3FF", "pass-through": "#F0F0F0"}

        # Create subgraphs for each execution level
        for i, level in enumerate(self.execution_levels):
            with dot.subgraph(name=f"cluster_{i}") as c:
                c.attr(label=f"Execution Level {i + 1}", style="filled", color="lightgrey")

                for req_name in level:
                    req = next(r for r in self.requirements if r.name == req_name)
                    eval_color = evaluator_colors.get(req.evaluator.type, "#FFFFFF")

                    # Shorter labels to fit better
                    label = f"{req.name}\\n"
                    label += f"E:{req.evaluator.name[:10]}\\n"
                    label += f"S:{req.query.scoring_config.subtype[:10]}\\n"
                    label += f"A:{req.aggregator.name[:10]}"

                    c.node(req.name, label, fillcolor=eval_color, fontsize="8")

        # Add dependency edges
        for req_name, deps in self.dependencies.items():
            if deps:
                for dep in deps:
                    dot.edge(dep, req_name, color="darkblue", penwidth="2")

        if save_path:
            dot.render(save_path, format="png", cleanup=True)
            print(f"Graphviz diagram saved to {save_path}.png")

        src: str = dot.source
        return src

    def create_plotly_interactive_graph(self, save_path: t.Optional[str] = None) -> go.Figure:
        """Create clean interactive Plotly graph with component reference panel."""
        G = self.create_networkx_graph()
        components = self.get_component_summary()
        completion = self.get_completion_summary()

        # Create layout based on execution levels with more spacing
        pos = {}
        level_height = 3.0  # Increased spacing
        level_width = 8.0  # Increased width

        for i, level in enumerate(self.execution_levels):
            y_pos = (len(self.execution_levels) - i - 1) * level_height
            n_nodes = len(level)

            for j, node in enumerate(level):
                if n_nodes == 1:
                    x_pos = 0.0
                else:
                    x_pos = (j - (n_nodes - 1) / 2) * (level_width / max(n_nodes - 1, 1))
                pos[node] = (x_pos, y_pos)

        # Extract edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=3, color="darkblue"),
            hoverinfo="none",
            mode="lines",
            name="Dependencies",
            showlegend=True,
        )

        # Extract nodes by completion status
        traces = [edge_trace]
        completion_colors = {
            "completed": "#90EE90",  # Light green
            "answered": "#FFE4B5",  # Light yellow/orange
            "pending": "#FFB6C1",  # Light pink
        }

        for status in ["completed", "answered", "pending"]:
            node_x = []
            node_y = []
            node_text = []
            node_info = []

            for node in G.nodes():
                if G.nodes[node]["completion_status"] == status:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                    # Add completion emoji to node display
                    completion_emoji = {"completed": "✓", "answered": "○", "pending": "-"}[status]
                    node_text.append(f"{completion_emoji} {node}")

                    # Create hover info with completion details
                    attrs = G.nodes[node]
                    info = f"<b>{completion_emoji} {node}</b><br>"
                    info += f"<b>Status:</b> {status.title()}<br>"

                    if attrs["score"] is not None:
                        score_info = f"{attrs['score']}"
                        if attrs["confidence"] is not None:
                            score_info += f" (confidence: {attrs['confidence']:.2f})"
                        info += f"<b>Score:</b> {score_info}<br>"

                    info += f"<b>Execution Level:</b> {attrs['execution_level']}<br>"
                    info += f"<b>Evaluator:</b> {attrs['evaluator_name']} ({attrs['evaluator_type']})<br>"
                    info += f"<b>Scoring:</b> {attrs['scoring_type']}<br>"
                    info += f"<b>Aggregator:</b> {attrs['aggregator_name']}<br>"
                    info += f"<b>Instruction:</b> {attrs['instruction'][:100]}{'...' if len(attrs['instruction']) > 100 else ''}"
                    node_info.append(info)

            if node_x:  # Only add trace if there are nodes of this status
                status_emoji = {"completed": "✓", "answered": "○", "pending": "-"}[status]
                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=node_text,
                    textposition="middle center",
                    textfont=dict(size=8, color="black"),
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=node_info,
                    marker=dict(
                        size=80,  # Much larger markers
                        color=completion_colors[status],
                        line=dict(width=2, color="darkblue"),
                    ),
                    name=f"{status_emoji} {status.title()} ({len(node_x)})",
                    showlegend=True,
                )
                traces.append(node_trace)

        # Create subtitle with components reference and completion status
        evaluators_text = self._format_component_set(components["evaluators"])
        scoring_text = self._format_component_set(components["scoring_configs"])
        aggregators_text = self._format_component_set(components["aggregators"])

        subtitle = f"✓ Completed: {completion['completed']}, ○ Answered: {completion['answered']}, - Pending: {completion['pending']} / {completion['total']}<br>"
        subtitle += f"• <b>Evaluators:</b> {evaluators_text}<br>"
        subtitle += f"• <b>Scoring:</b> {scoring_text}<br>"
        subtitle += f"• <b>Aggregators:</b> {aggregators_text}"

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(
                    text=f"Rubric Architecture with Completion Status<br><sub>{subtitle}</sub>",
                    font=dict(size=16),
                ),
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=250),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive graph saved to {save_path}")

        fig.show()
        return fig


def visualize_rubric(
    rubric: t.Any, output_dir: t.Optional[str] = None, inputs: t.Optional[str] = None
) -> tuple[RubricVisualizer, Rubric]:
    if isinstance(rubric, str) and rubric.endswith(".yaml"):
        rubric = Rubric.from_yaml(rubric)
    elif isinstance(rubric, Rubric):
        pass
    else:
        raise ValueError(f"Invalid rubric type: {type(rubric)}")

    if inputs:
        rubric.solve(inputs)
    visualizer = RubricVisualizer(rubric)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate all available visualizations
        visualizer.plot_networkx_graph(str(output_path / "networkx_graph.png"))
        visualizer.create_graphviz_diagram(str(output_path / "graphviz_diagram.png"))
        visualizer.create_plotly_interactive_graph(
            str(output_path / "plotly_interactive_graph.html")
        )

    return visualizer, rubric


def quick_graph_viz(
    rubric_path: str, save_path: t.Optional[str] = None, inputs: t.Optional[str] = None
) -> None:
    """Quick graph visualization."""
    rubric = Rubric.from_yaml(rubric_path)
    if inputs:
        rubric.solve(inputs)
    visualizer = RubricVisualizer(rubric)
    visualizer.plot_networkx_graph(save_path)


def quick_interactive_viz(
    rubric_path: str, save_path: t.Optional[str] = None, inputs: t.Optional[str] = None
) -> None:
    """Quick interactive visualization."""
    rubric = Rubric.from_yaml(rubric_path)
    if inputs:
        rubric.solve(inputs)
    visualizer = RubricVisualizer(rubric)
    visualizer.create_plotly_interactive_graph(save_path)
