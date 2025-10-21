"""
Visual Data Explorer for SLM Personal Agent
Advanced data visualization, charting, and interactive graph generation
"""

import asyncio
import json
import logging
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import io
import base64
import os
from pathlib import Path

from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.asyncio import AsyncSession

from .database import Base, memory_manager
from .ollama_client import call_ollama

logger = logging.getLogger(__name__)

class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar" 
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    AREA = "area"
    RADAR = "radar"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    NETWORK = "network"
    TIMELINE = "timeline"

class DataSource(str, Enum):
    UPLOADED = "uploaded"
    CONVERSATIONS = "conversations"
    INTEGRATIONS = "integrations"
    SECURITY_LOGS = "security_logs"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    GENERATED = "generated"

@dataclass
class VisualizationConfig:
    chart_type: ChartType
    title: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    facet_column: Optional[str] = None
    aggregation: Optional[str] = None  # sum, count, mean, etc.
    theme: str = "plotly"
    width: int = 800
    height: int = 600
    interactive: bool = True
    animation_frame: Optional[str] = None

class DataVisualization(Base):
    __tablename__ = "data_visualizations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Data source
    data_source = Column(String, nullable=False)  # DataSource enum
    data_query = Column(Text)  # SQL query or API call
    data_file_path = Column(String)  # For uploaded files
    
    # Visualization config
    chart_type = Column(String, nullable=False)  # ChartType enum
    config = Column(JSON)  # VisualizationConfig as JSON
    
    # Generated outputs
    static_image_path = Column(String)  # PNG/SVG file path
    interactive_html_path = Column(String)  # HTML file path
    plotly_json = Column(Text)  # Plotly JSON for embedding
    
    # Metadata
    data_rows = Column(Integer, default=0)
    data_columns = Column(Integer, default=0)
    file_size = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)

class DataExplorerEngine:
    def __init__(self):
        self.engine = memory_manager.engine
        self.async_session = memory_manager.async_session
        self._initialized = False
        self.output_dir = Path("static/visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting styles
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure Plotly
        pio.templates.default = "plotly_white"
        
    async def initialize(self):
        """Initialize the data explorer"""
        if self._initialized:
            return
            
        # Create visualization tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._initialized = True
        logger.info("Visual Data Explorer initialized")
    
    async def analyze_data(self, data: pd.DataFrame, data_source: str = "uploaded") -> Dict[str, Any]:
        """Analyze dataset and suggest visualizations"""
        try:
            analysis = {
                "dataset_info": {
                    "rows": len(data),
                    "columns": len(data.columns),
                    "size_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
                    "column_types": data.dtypes.to_dict()
                },
                "column_analysis": {},
                "suggested_visualizations": [],
                "data_quality": {}
            }
            
            # Analyze each column
            for col in data.columns:
                col_analysis = {
                    "type": str(data[col].dtype),
                    "null_count": data[col].isnull().sum(),
                    "null_percentage": (data[col].isnull().sum() / len(data)) * 100,
                    "unique_values": data[col].nunique()
                }
                
                if data[col].dtype in ['int64', 'float64']:
                    col_analysis.update({
                        "mean": data[col].mean(),
                        "median": data[col].median(),
                        "std": data[col].std(),
                        "min": data[col].min(),
                        "max": data[col].max(),
                        "quartiles": data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                    })
                elif data[col].dtype == 'object':
                    col_analysis.update({
                        "most_common": data[col].value_counts().head(5).to_dict(),
                        "average_length": data[col].astype(str).str.len().mean() if not data[col].empty else 0
                    })
                
                analysis["column_analysis"][col] = col_analysis
            
            # Data quality assessment
            analysis["data_quality"] = {
                "overall_completeness": ((data.size - data.isnull().sum().sum()) / data.size) * 100,
                "duplicate_rows": data.duplicated().sum(),
                "columns_with_nulls": data.isnull().any().sum(),
                "recommendations": []
            }
            
            # Add recommendations
            if analysis["data_quality"]["duplicate_rows"] > 0:
                analysis["data_quality"]["recommendations"].append(
                    f"Consider removing {analysis['data_quality']['duplicate_rows']} duplicate rows"
                )
            
            high_null_cols = [col for col, info in analysis["column_analysis"].items() 
                            if info["null_percentage"] > 50]
            if high_null_cols:
                analysis["data_quality"]["recommendations"].append(
                    f"Columns with >50% nulls: {', '.join(high_null_cols)}"
                )
            
            # Suggest visualizations based on data types
            numeric_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
            categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
            datetime_cols = [col for col in data.columns if data[col].dtype.name.startswith('datetime')]
            
            # Suggest visualizations
            if len(numeric_cols) >= 2:
                analysis["suggested_visualizations"].append({
                    "type": "scatter",
                    "title": f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}",
                    "config": {
                        "x_column": numeric_cols[0],
                        "y_column": numeric_cols[1],
                        "color_column": categorical_cols[0] if categorical_cols else None
                    }
                })
            
            if len(numeric_cols) >= 1:
                analysis["suggested_visualizations"].append({
                    "type": "histogram",
                    "title": f"Distribution of {numeric_cols[0]}",
                    "config": {"x_column": numeric_cols[0]}
                })
            
            if len(categorical_cols) >= 1:
                analysis["suggested_visualizations"].append({
                    "type": "bar",
                    "title": f"Count by {categorical_cols[0]}",
                    "config": {
                        "x_column": categorical_cols[0],
                        "aggregation": "count"
                    }
                })
            
            if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
                analysis["suggested_visualizations"].append({
                    "type": "line",
                    "title": f"Trend of {numeric_cols[0]} over time",
                    "config": {
                        "x_column": datetime_cols[0],
                        "y_column": numeric_cols[0]
                    }
                })
            
            if len(numeric_cols) >= 3:
                analysis["suggested_visualizations"].append({
                    "type": "heatmap",
                    "title": "Correlation Heatmap",
                    "config": {"correlation": True}
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Data analysis error: {e}")
            raise ValueError(f"Failed to analyze data: {e}")
    
    async def create_visualization(self, data: pd.DataFrame, config: VisualizationConfig, 
                                 name: str, description: str = "", 
                                 data_source: DataSource = DataSource.UPLOADED) -> str:
        """Create a visualization and save it"""
        await self.initialize()
        
        try:
            # Generate visualization
            if config.interactive:
                fig, plotly_json = await self._create_plotly_chart(data, config)
                html_path = self.output_dir / f"{uuid.uuid4()}.html"
                fig.write_html(str(html_path))
                static_path = None
            else:
                fig = await self._create_matplotlib_chart(data, config)
                static_path = self.output_dir / f"{uuid.uuid4()}.png"
                fig.savefig(str(static_path), dpi=300, bbox_inches='tight')
                plt.close(fig)
                html_path = None
                plotly_json = None
            
            # Save to database
            visualization = DataVisualization(
                name=name,
                description=description,
                data_source=data_source.value,
                chart_type=config.chart_type.value,
                config=asdict(config),
                static_image_path=str(static_path) if static_path else None,
                interactive_html_path=str(html_path) if html_path else None,
                plotly_json=plotly_json,
                data_rows=len(data),
                data_columns=len(data.columns),
                file_size=data.memory_usage(deep=True).sum()
            )
            
            async with self.async_session() as session:
                session.add(visualization)
                await session.commit()
            
            return visualization.id
            
        except Exception as e:
            logger.error(f"Visualization creation error: {e}")
            raise ValueError(f"Failed to create visualization: {e}")
    
    async def _create_plotly_chart(self, data: pd.DataFrame, config: VisualizationConfig) -> Tuple[go.Figure, str]:
        """Create interactive Plotly chart"""
        fig = None
        
        # Auto-select columns if not specified
        if not config.x_column or not config.y_column:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Set default columns based on chart type and available data
            if config.chart_type == ChartType.LINE:
                config.x_column = config.x_column or (datetime_cols[0] if datetime_cols else (categorical_cols[0] if categorical_cols else data.columns[0]))
                config.y_column = config.y_column or (numeric_cols[0] if numeric_cols else data.columns[1])
            elif config.chart_type == ChartType.BAR:
                config.x_column = config.x_column or (categorical_cols[0] if categorical_cols else data.columns[0])
                config.y_column = config.y_column or (numeric_cols[0] if numeric_cols else data.columns[1])
            else:
                config.x_column = config.x_column or data.columns[0]
                config.y_column = config.y_column or (data.columns[1] if len(data.columns) > 1 else data.columns[0])
        
        # Ensure data types are appropriate for the chart
        try:
            if config.chart_type == ChartType.LINE:
                # Convert date column if it exists
                if config.x_column in data.columns and data[config.x_column].dtype == 'object':
                    try:
                        data[config.x_column] = pd.to_datetime(data[config.x_column])
                    except:
                        pass  # Keep as is if conversion fails
                        
                fig = px.line(data, x=config.x_column, y=config.y_column, 
                             color=config.color_column, title=config.title,
                             animation_frame=config.animation_frame)
        except Exception as e:
            logger.warning(f"Line chart creation failed: {e}, falling back to scatter plot")
            fig = px.scatter(data, x=config.x_column, y=config.y_column, 
                           color=config.color_column, title=config.title)
        
        if not fig:  # If line chart creation failed or not a line chart
            try:
                if config.chart_type == ChartType.BAR:
                    if config.aggregation:
                        # Aggregate data first
                        if config.aggregation == "count":
                            agg_data = data.groupby(config.x_column).size().reset_index(name='count')
                            fig = px.bar(agg_data, x=config.x_column, y='count', title=config.title)
                        else:
                            agg_data = data.groupby(config.x_column)[config.y_column].agg(config.aggregation).reset_index()
                            fig = px.bar(agg_data, x=config.x_column, y=config.y_column, title=config.title)
                    else:
                        fig = px.bar(data, x=config.x_column, y=config.y_column, 
                                   color=config.color_column, title=config.title)
                
                elif config.chart_type == ChartType.SCATTER:
                    fig = px.scatter(data, x=config.x_column, y=config.y_column,
                                   color=config.color_column, size=config.size_column,
                                   title=config.title, animation_frame=config.animation_frame)
                
                elif config.chart_type == ChartType.PIE:
                    if config.aggregation == "count":
                        pie_data = data[config.x_column].value_counts().reset_index()
                        fig = px.pie(pie_data, values=config.x_column, names='index', title=config.title)
                    else:
                        fig = px.pie(data, values=config.y_column, names=config.x_column, title=config.title)
                
                elif config.chart_type == ChartType.HISTOGRAM:
                    fig = px.histogram(data, x=config.x_column, color=config.color_column,
                                     title=config.title, nbins=30)
                
                elif config.chart_type == ChartType.HEATMAP:
                    if config.config.get("correlation", False):
                        # Create correlation heatmap
                        numeric_data = data.select_dtypes(include=[np.number])
                        corr_matrix = numeric_data.corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                      title="Correlation Heatmap")
                    else:
                        # Regular heatmap
                        fig = px.density_heatmap(data, x=config.x_column, y=config.y_column,
                                               title=config.title)
                
                elif config.chart_type == ChartType.BOX:
                    fig = px.box(data, x=config.x_column, y=config.y_column,
                                color=config.color_column, title=config.title)
                
                elif config.chart_type == ChartType.VIOLIN:
                    fig = px.violin(data, x=config.x_column, y=config.y_column,
                                  color=config.color_column, title=config.title)
                
                elif config.chart_type == ChartType.AREA:
                    fig = px.area(data, x=config.x_column, y=config.y_column,
                                 color=config.color_column, title=config.title)
                
                elif config.chart_type == ChartType.TREEMAP:
                    fig = px.treemap(data, path=[px.Constant("All"), config.x_column],
                                   values=config.y_column, title=config.title)
                
                else:
                    # Default to scatter plot
                    fig = px.scatter(data, x=config.x_column, y=config.y_column, title=config.title)
                    
            except Exception as e:
                logger.warning(f"Chart creation failed: {e}, creating fallback scatter plot")
                # Fallback to basic scatter plot with first two columns
                fig = px.scatter(data, x=data.columns[0], 
                               y=data.columns[1] if len(data.columns) > 1 else data.columns[0], 
                               title=config.title or "Data Visualization")
        
        # Apply styling
        fig.update_layout(
            width=config.width,
            height=config.height,
            template=config.theme if config.theme in pio.templates else "plotly_white"
        )
        
        # Convert to JSON for storage
        plotly_json = fig.to_json()
        
        return fig, plotly_json
    
    async def _create_matplotlib_chart(self, data: pd.DataFrame, config: VisualizationConfig) -> plt.Figure:
        """Create static matplotlib chart"""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        if config.chart_type == ChartType.LINE:
            data.plot(x=config.x_column, y=config.y_column, kind='line', ax=ax)
        
        elif config.chart_type == ChartType.BAR:
            if config.aggregation == "count":
                data[config.x_column].value_counts().plot(kind='bar', ax=ax)
            else:
                data.plot(x=config.x_column, y=config.y_column, kind='bar', ax=ax)
        
        elif config.chart_type == ChartType.SCATTER:
            ax.scatter(data[config.x_column], data[config.y_column])
        
        elif config.chart_type == ChartType.HISTOGRAM:
            data[config.x_column].plot(kind='hist', ax=ax, bins=30)
        
        elif config.chart_type == ChartType.HEATMAP:
            if config.config.get("correlation", False):
                numeric_data = data.select_dtypes(include=[np.number])
                sns.heatmap(numeric_data.corr(), annot=True, ax=ax)
            else:
                # Pivot data for heatmap
                pivot_data = data.pivot_table(values=config.y_column, 
                                            index=config.x_column, 
                                            columns=config.color_column, 
                                            aggfunc='mean')
                sns.heatmap(pivot_data, ax=ax)
        
        elif config.chart_type == ChartType.BOX:
            data.boxplot(column=config.y_column, by=config.x_column, ax=ax)
        
        else:
            # Default scatter
            ax.scatter(data[config.x_column], data[config.y_column])
        
        ax.set_title(config.title)
        plt.tight_layout()
        
        return fig
    
    async def generate_sample_data(self, data_type: str = "sales", rows: int = 100) -> pd.DataFrame:
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        if data_type == "sales":
            dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
            data = pd.DataFrame({
                'date': dates,
                'sales': np.random.normal(1000, 200, rows),
                'region': np.random.choice(['North', 'South', 'East', 'West'], rows),
                'product': np.random.choice(['Product A', 'Product B', 'Product C'], rows),
                'profit_margin': np.random.uniform(0.1, 0.4, rows)
            })
            # Make sales positive
            data['sales'] = np.abs(data['sales'])
            
        elif data_type == "users":
            data = pd.DataFrame({
                'age': np.random.randint(18, 65, rows),
                'gender': np.random.choice(['Male', 'Female', 'Other'], rows),
                'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany'], rows),
                'signup_date': pd.date_range(start='2023-01-01', periods=rows, freq='D'),
                'monthly_spend': np.random.exponential(50, rows),
                'satisfaction_score': np.random.uniform(1, 10, rows)
            })
            
        elif data_type == "stocks":
            dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
            base_price = 100
            data = pd.DataFrame({
                'date': dates,
                'price': base_price + np.cumsum(np.random.normal(0, 2, rows)),
                'volume': np.random.exponential(1000000, rows),
                'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], rows)
            })
            # Ensure positive prices
            data['price'] = np.abs(data['price'])
            
        else:
            # Generic data
            data = pd.DataFrame({
                'x': np.random.randn(rows),
                'y': np.random.randn(rows),
                'category': np.random.choice(['A', 'B', 'C'], rows),
                'value': np.random.uniform(0, 100, rows)
            })
        
        return data
    
    async def get_conversation_analytics(self) -> pd.DataFrame:
        """Get analytics data from conversations"""
        try:
            async with self.async_session() as session:
                from sqlalchemy import select, func
                from .database import Conversation, Message
                
                # Get conversation statistics
                conv_query = select(
                    Conversation.mode,
                    func.count(Conversation.id).label('conversation_count'),
                    func.avg(func.julianday(Conversation.updated_at) - func.julianday(Conversation.created_at)).label('avg_duration_days')
                ).group_by(Conversation.mode)
                
                conv_result = await session.execute(conv_query)
                conv_data = conv_result.fetchall()
                
                # Get message statistics
                msg_query = select(
                    Message.mode,
                    func.count(Message.id).label('message_count'),
                    func.avg(func.length(Message.content)).label('avg_message_length'),
                    func.date(Message.created_at).label('date')
                ).group_by(Message.mode, func.date(Message.created_at))
                
                msg_result = await session.execute(msg_query)
                msg_data = msg_result.fetchall()
                
                # Convert to DataFrame
                if msg_data:
                    df = pd.DataFrame(msg_data, columns=['mode', 'message_count', 'avg_message_length', 'date'])
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                else:
                    # Return sample data if no conversations exist
                    return await self.generate_sample_data("conversations", 50)
                    
        except Exception as e:
            logger.error(f"Conversation analytics error: {e}")
            # Return sample data on error
            return await self.generate_sample_data("conversations", 50)
    
    async def get_saved_visualizations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of saved visualizations"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select, desc
            
            stmt = select(DataVisualization).order_by(desc(DataVisualization.updated_at)).limit(limit)
            result = await session.execute(stmt)
            visualizations = result.scalars().all()
            
            return [
                {
                    "id": viz.id,
                    "name": viz.name,
                    "description": viz.description,
                    "chart_type": viz.chart_type,
                    "data_source": viz.data_source,
                    "data_rows": viz.data_rows,
                    "data_columns": viz.data_columns,
                    "created_at": viz.created_at.isoformat(),
                    "has_interactive": viz.interactive_html_path is not None,
                    "has_static": viz.static_image_path is not None
                }
                for viz in visualizations
            ]
    
    async def get_visualization(self, viz_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific visualization"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(DataVisualization).where(DataVisualization.id == viz_id)
            result = await session.execute(stmt)
            viz = result.scalar_one_or_none()
            
            if not viz:
                return None
            
            # Update last accessed
            viz.last_accessed = datetime.utcnow()
            await session.commit()
            
            viz_data = {
                "id": viz.id,
                "name": viz.name,
                "description": viz.description,
                "chart_type": viz.chart_type,
                "data_source": viz.data_source,
                "config": viz.config,
                "data_rows": viz.data_rows,
                "data_columns": viz.data_columns,
                "created_at": viz.created_at.isoformat(),
                "updated_at": viz.updated_at.isoformat(),
                "plotly_json": viz.plotly_json,
                "static_image_path": viz.static_image_path,
                "interactive_html_path": viz.interactive_html_path
            }
            
            return viz_data
    
    async def create_dashboard(self, visualization_ids: List[str], 
                             title: str = "Data Dashboard") -> str:
        """Create a dashboard combining multiple visualizations"""
        try:
            # Get visualizations
            visualizations = []
            for viz_id in visualization_ids:
                viz = await self.get_visualization(viz_id)
                if viz:
                    visualizations.append(viz)
            
            if not visualizations:
                raise ValueError("No valid visualizations found")
            
            # Create dashboard HTML
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .dashboard-header {{ text-align: center; margin-bottom: 30px; }}
                    .viz-container {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                    .viz-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                    .viz-description {{ color: #666; margin-bottom: 15px; }}
                    .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                </style>
            </head>
            <body>
                <div class="dashboard-header">
                    <h1>{title}</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="dashboard-grid">
            """
            
            for i, viz in enumerate(visualizations):
                dashboard_html += f"""
                    <div class="viz-container">
                        <div class="viz-title">{viz['name']}</div>
                        <div class="viz-description">{viz['description']}</div>
                        <div id="chart-{i}"></div>
                    </div>
                """
            
            dashboard_html += """
                </div>
                
                <script>
            """
            
            for i, viz in enumerate(visualizations):
                if viz['plotly_json']:
                    dashboard_html += f"""
                        var chartData_{i} = {viz['plotly_json']};
                        Plotly.newPlot('chart-{i}', chartData_{i}.data, chartData_{i}.layout);
                    """
            
            dashboard_html += """
                </script>
            </body>
            </html>
            """
            
            # Save dashboard
            dashboard_path = self.output_dir / f"dashboard_{uuid.uuid4()}.html"
            with open(dashboard_path, 'w') as f:
                f.write(dashboard_html)
            
            return str(dashboard_path)
            
        except Exception as e:
            logger.error(f"Dashboard creation error: {e}")
            raise ValueError(f"Failed to create dashboard: {e}")

# Global data explorer instance
data_explorer = DataExplorerEngine()