import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import graphviz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from open_rubric.rubric import Rubric
from open_rubric.configs.requirement import RequirementConfig
from open_rubric.configs.evaluating import BaseEvaluatorConfig, ModelEvaluatorConfig, EnsembledModelEvaluatorConfig
from open_rubric.configs.aggregating import BaseAggregatingConfig
from open_rubric.configs.scoring import ScoringConfig
from open_rubric.utils.dag import topological_levels


VIZ_OUTPUT_DIR = Path("assets/viz_outputs/")

class RubricVisualizer:
    """Comprehensive visualization toolkit for Open Rubric system."""
    
    def __init__(self, rubric: Rubric):
        self.rubric = rubric
        self.requirements = rubric.requirements.get_all_requirements()
        self.dependencies = rubric.requirements.dependencies
        self.execution_levels = topological_levels(self.dependencies)

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
            
            G.add_node(
                req.name,
                node_type='requirement',
                title=req.name,
                execution_level=execution_level,
                evaluator_name=req.evaluator.name,
                evaluator_type=req.evaluator.type,
                scoring_type=req.query.scoring_config.subtype,
                aggregator_name=req.aggregator.name,
                instruction=req.query.instruction,
                full_instruction=req.query.instruction
            )
        
        # Add edges for requirement dependencies only
        for req_name, deps in self.dependencies.items():
            if deps:
                for dep in deps:
                    G.add_edge(dep, req_name, edge_type='dependency')
        
        return G

    def get_component_summary(self) -> dict[str, set]:
        """Get summary of all components used in the rubric."""
        evaluators = set()
        scoring_configs = set()
        aggregators = set()
        
        for req in self.requirements:
            evaluators.add((req.evaluator.name, req.evaluator.type))
            scoring_configs.add(req.query.scoring_config.subtype)
            aggregators.add(req.aggregator.name)
        
        return {
            'evaluators': evaluators,
            'scoring_configs': scoring_configs,
            'aggregators': aggregators
        }
    
    def plot_networkx_graph(self, save_path: t.Optional[str] = None, figsize: tuple[int, int] = (14, 10)) -> None:
        """Plot clean NetworkX graph with requirements and component reference panel."""
        G = self.create_networkx_graph()
        components = self.get_component_summary()
        
        fig, (ax_components, ax_graph) = plt.subplots(2, 1, figsize=figsize, 
                                                     gridspec_kw={'height_ratios': [1, 4]})
        
        # Top panel: Component reference
        ax_components.text(0.02, 0.8, "EVALUATORS:", fontweight='bold', fontsize=10, transform=ax_components.transAxes)
        eval_text = ", ".join([f"{name} ({type_})" for name, type_ in components['evaluators']])
        ax_components.text(0.02, 0.6, eval_text, fontsize=9, transform=ax_components.transAxes, wrap=True)
        
        ax_components.text(0.02, 0.4, "SCORING:", fontweight='bold', fontsize=10, transform=ax_components.transAxes)
        scoring_text = ", ".join(components['scoring_configs'])
        ax_components.text(0.02, 0.2, scoring_text, fontsize=9, transform=ax_components.transAxes)
        
        ax_components.text(0.5, 0.4, "AGGREGATORS:", fontweight='bold', fontsize=10, transform=ax_components.transAxes)
        agg_text = ", ".join(components['aggregators'])
        ax_components.text(0.5, 0.2, agg_text, fontsize=9, transform=ax_components.transAxes)
        
        ax_components.set_xlim(0, 1)
        ax_components.set_ylim(0, 1)
        ax_components.axis('off')
        ax_components.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.3))
        
        # Bottom panel: Requirements graph
        plt.sca(ax_graph)
        
        # Create hierarchical layout based on execution levels
        pos = {}
        level_height = 2.0
        level_width = 6.0
        
        for i, level in enumerate(self.execution_levels):
            y_pos = (len(self.execution_levels) - i - 1) * level_height
            n_nodes = len(level)
            
            for j, node in enumerate(level):
                if n_nodes == 1:
                    x_pos = 0
                else:
                    x_pos = (j - (n_nodes - 1) / 2) * (level_width / max(n_nodes - 1, 1))
                pos[node] = (x_pos, y_pos)
        
        # Color nodes by evaluator type for easy identification
        evaluator_types = list(set(G.nodes[node]['evaluator_type'] for node in G.nodes()))
        colors = plt.cm.Set3(range(len(evaluator_types)))
        evaluator_color_map = dict(zip(evaluator_types, colors))
        
        node_colors = [evaluator_color_map[G.nodes[node]['evaluator_type']] for node in G.nodes()]
        
        # Draw execution level backgrounds
        for i, level in enumerate(self.execution_levels):
            y_pos = (len(self.execution_levels) - i - 1) * level_height
            rect = plt.Rectangle((-level_width/2 - 0.5, y_pos - 0.4), 
                               level_width + 1, 0.8, 
                               facecolor='lightgray', 
                               alpha=0.2, 
                               edgecolor='gray')
            ax_graph.add_patch(rect)
            ax_graph.text(-level_width/2 - 0.3, y_pos, f'Level {i + 1}', 
                    rotation=90, va='center', ha='center', fontweight='bold')
        
        # Draw graph
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='darkblue',
                width=2,
                alpha=0.8,
                ax=ax_graph)
        
        # Add requirement labels with component info
        labels = {}
        for node in G.nodes():
            attrs = G.nodes[node]
            label = f"{attrs['title']}\n"
            label += f"E:{attrs['evaluator_name']} S:{attrs['scoring_type']}\nA:{attrs['aggregator_name']}"
            labels[node] = label
        
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax_graph)
        
        # Create legend for evaluator types
        legend_patches = [mpatches.Patch(color=color, label=eval_type) 
                         for eval_type, color in evaluator_color_map.items()]
        ax_graph.legend(handles=legend_patches, title="Evaluator Types", loc='upper left', bbox_to_anchor=(1, 1))
        
        ax_graph.set_title("Requirements Dependencies\n(Organized by Execution Levels)", fontsize=14, fontweight='bold')
        ax_graph.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved to {save_path}")
        
    
    def create_graphviz_diagram(self, save_path: t.Optional[str] = None) -> str:
        """Create clean Graphviz diagram with component reference."""
        dot = graphviz.Digraph(comment='Clean Rubric Structure')
        dot.attr(rankdir='TB', size='12,10!')
        dot.attr('node', shape='box', style='rounded,filled')
        
        components = self.get_component_summary()
        
        # Add component reference at top
        ref_label = "COMPONENTS REFERENCE\\n\\n"
        ref_label += "EVALUATORS: " + ", ".join([f"{name}({type_})" for name, type_ in components['evaluators']]) + "\\n"
        ref_label += "SCORING: " + ", ".join(components['scoring_configs']) + "\\n"
        ref_label += "AGGREGATORS: " + ", ".join(components['aggregators'])
        
        dot.node('reference', ref_label, shape='note', fillcolor='lightgray', fontsize='10')
        
        # Define colors for different evaluator types
        evaluator_types = list(set(req.evaluator.type for req in self.requirements))
        evaluator_colors = {
            'llm': '#FFE5E5',
            'llm-ensemble': '#E5F3FF', 
            'pass-through': '#F0F0F0'
        }
        
        # Create subgraphs for each execution level
        for i, level in enumerate(self.execution_levels):
            with dot.subgraph(name=f'cluster_{i}') as c:
                c.attr(label=f'Execution Level {i + 1}', style='filled', color='lightgrey')
                
                for req_name in level:
                    req = next(r for r in self.requirements if r.name == req_name)
                    eval_color = evaluator_colors.get(req.evaluator.type, '#FFFFFF')
                    
                    label = f"{req.name}\\n"
                    label += f"E:{req.evaluator.name}\\n"
                    label += f"S:{req.query.scoring_config.subtype}\\n"
                    label += f"A:{req.aggregator.name}"
                    
                    c.node(req.name, label, fillcolor=eval_color, fontsize='9')
        
        # Add dependency edges
        for req_name, deps in self.dependencies.items():
            if deps:
                for dep in deps:
                    dot.edge(dep, req_name, color='darkblue', penwidth='2')
        
        if save_path:
            dot.render(save_path, format='png', cleanup=True)
            print(f"Graphviz diagram saved to {save_path}.png")
        
        return dot.source
    
    def create_plotly_interactive_graph(self, save_path: t.Optional[str] = None) -> go.Figure:
        """Create clean interactive Plotly graph with component reference panel."""
        G = self.create_networkx_graph()
        components = self.get_component_summary()
        
        # Create layout based on execution levels
        pos = {}
        level_height = 2.0
        level_width = 6.0
        
        for i, level in enumerate(self.execution_levels):
            y_pos = (len(self.execution_levels) - i - 1) * level_height
            n_nodes = len(level)
            
            for j, node in enumerate(level):
                if n_nodes == 1:
                    x_pos = 0
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
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=3, color='darkblue'),
                               hoverinfo='none',
                               mode='lines',
                               name='Dependencies',
                               showlegend=True)
        
        # Extract nodes by execution level
        traces = [edge_trace]
        colors = px.colors.qualitative.Set3
        
        for i, level in enumerate(self.execution_levels):
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            
            for node in level:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Create hover info
                attrs = G.nodes[node]
                info = f"<b>{node}</b><br>"
                info += f"<b>Execution Level:</b> {i + 1}<br>"
                info += f"<b>Evaluator:</b> {attrs['evaluator_name']} ({attrs['evaluator_type']})<br>"
                info += f"<b>Scoring:</b> {attrs['scoring_type']}<br>"
                info += f"<b>Aggregator:</b> {attrs['aggregator_name']}<br>"
                info += f"<b>Instruction:</b> {attrs['instruction'][:100]}{'...' if len(attrs['instruction']) > 100 else ''}"
                node_info.append(info)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hovertemplate='%{text}<extra></extra>',
                text=node_info,
                textposition="middle center",
                marker=dict(size=50,
                           color=colors[i % len(colors)],
                           line=dict(width=2, color='darkblue')),
                name=f'Level {i + 1}',
                showlegend=True
            )
            traces.append(node_trace)
        
        # Create component reference text
        ref_text = "COMPONENTS REFERENCE<br><br>"
        ref_text += "<b>EVALUATORS:</b> " + ", ".join([f"{name} ({type_})" for name, type_ in components['evaluators']]) + "<br>"
        ref_text += "<b>SCORING:</b> " + ", ".join(components['scoring_configs']) + "<br>"
        ref_text += "<b>AGGREGATORS:</b> " + ", ".join(components['aggregators'])
        
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title=dict(
                               text='Clean Rubric Architecture<br><sub>Hover over nodes for details • Colors represent execution levels</sub>',
                               font=dict(size=16)
                           ),
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=100),
                           annotations=[
                               dict(
                                   text=ref_text,
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.02, y=0.98,
                                   xanchor="left", yanchor="top",
                                   font=dict(color="#000000", size=10),
                                   bgcolor="rgba(240,240,240,0.8)",
                                   bordercolor="gray",
                                   borderwidth=1
                               )
                           ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive graph saved to {save_path}")
        
        fig.show()
        return fig


def visualize_rubric_from_path(rubric_path: str, output_dir: t.Optional[str] = None) -> RubricVisualizer:
    """Convenience function to create visualizations from a rubric YAML file."""
    rubric = Rubric.from_yaml(rubric_path)
    visualizer = RubricVisualizer(rubric)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all available visualizations
        visualizer.plot_networkx_graph(str(output_path / "dependency_graph.png"))
        visualizer.create_graphviz_diagram(str(output_path / "graphviz_diagram"))
        visualizer.create_plotly_interactive_graph(str(output_path / "interactive_graph.html"))
    
    return visualizer


def quick_graph_viz(rubric_path: str, save_path: t.Optional[str] = None) -> None:
    """Quick graph visualization."""
    rubric = Rubric.from_yaml(rubric_path)
    visualizer = RubricVisualizer(rubric)
    visualizer.plot_networkx_graph(save_path)

def quick_interactive_viz(rubric_path: str, save_path: t.Optional[str] = None) -> None:
    """Quick interactive visualization."""
    rubric = Rubric.from_yaml(rubric_path)
    visualizer = RubricVisualizer(rubric)
    visualizer.create_plotly_interactive_graph(save_path)
