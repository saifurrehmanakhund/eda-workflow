import logging
import math
import os
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)
        
        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }

    def detect_duplicates_node(state: EDAState):
        """Detect duplicate records and data quality red flags."""
        logger.info("Detecting duplicates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows else 0

        id_columns = [col for col in df.columns if "id" in col.lower()]
        duplicate_ids = {}
        for col in id_columns:
            if df[col].isnull().all():
                continue
            dup_count = df[col].duplicated().sum()
            duplicate_ids[col] = {
                "duplicate_count": int(dup_count),
                "duplicate_pct": round((dup_count / total_rows) * 100, 2) if total_rows else 0,
            }

        red_flags = []
        if duplicate_pct > 1:
            red_flags.append("High duplicate row percentage (>1%)")
        for col, stats in duplicate_ids.items():
            if stats["duplicate_pct"] > 1:
                red_flags.append(f"High duplicate IDs in {col} (>1%)")

        duplicates = {
            "total_rows": total_rows,
            "duplicate_rows": int(duplicate_rows),
            "duplicate_pct": duplicate_pct,
            "duplicate_id_columns": duplicate_ids,
            "red_flags": red_flags,
        }

        results["detect_duplicates"] = duplicates

        return {
            "current_step": "detect_duplicates",
            "results": results,
        }

    def analyze_distributions_node(state: EDAState):
        """Analyze numeric distributions with skewness, kurtosis, and normality tests."""
        logger.info("Analyzing distributions")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        distributions = {}

        for col in numeric_cols:
            series = df[col].dropna()
            n = len(series)
            if n < 3:
                distributions[col] = {
                    "count": n,
                    "note": "Insufficient data for distribution analysis",
                }
                continue

            skewness = round(series.skew(), 4)
            kurtosis = round(series.kurt(), 4)

            # Jarque-Bera test (approx), chi-square df=2 => p = exp(-JB/2)
            jb_stat = (n / 6.0) * (skewness ** 2 + (kurtosis ** 2) / 4.0)
            jb_pvalue = round(math.exp(-jb_stat / 2.0), 6) if jb_stat >= 0 else 1.0

            distributions[col] = {
                "count": int(n),
                "skewness": skewness,
                "kurtosis": kurtosis,
                "jarque_bera": {
                    "statistic": round(jb_stat, 4),
                    "p_value": jb_pvalue,
                    "is_normal_approx": jb_pvalue >= 0.05,
                },
            }

        results["analyze_distributions"] = distributions if distributions else {
            "message": "No numeric columns available for distribution analysis"
        }

        return {
            "current_step": "analyze_distributions",
            "results": results,
        }

    def detect_outliers_node(state: EDAState):
        """Detect outliers using IQR and z-score methods."""
        logger.info("Detecting outliers")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        outliers = {}

        for col in numeric_cols:
            series = df[col].dropna()
            n = len(series)
            if n < 4:
                outliers[col] = {
                    "count": n,
                    "note": "Insufficient data for outlier detection",
                }
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_mask = (series < lower) | (series > upper)
            iqr_count = int(iqr_mask.sum())

            std = series.std()
            if std == 0 or pd.isna(std):
                z_count = 0
                z_sample = []
            else:
                z_scores = (series - series.mean()) / std
                z_mask = z_scores.abs() > 3
                z_count = int(z_mask.sum())
                z_sample = series[z_mask].head(5).round(4).tolist()

            outliers[col] = {
                "count": int(n),
                "iqr": {
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                    "outlier_count": iqr_count,
                    "outlier_pct": round((iqr_count / n) * 100, 2) if n else 0,
                },
                "z_score": {
                    "threshold": 3,
                    "outlier_count": z_count,
                    "outlier_pct": round((z_count / n) * 100, 2) if n else 0,
                    "sample_values": z_sample,
                },
            }

        results["detect_outliers"] = outliers if outliers else {
            "message": "No numeric columns available for outlier detection"
        }

        return {
            "current_step": "detect_outliers",
            "results": results,
        }
    
    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns."""
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        aggregates = {}
        
        # Compute aggregates grouped by each categorical column
        if categorical_cols and numeric_cols:
            for cat_col in categorical_cols:
                # Limit to top 10 categories to prevent token overflow
                top_categories = df[cat_col].value_counts().head(10).index
                df_subset = df[df[cat_col].isin(top_categories)]
                
                # Compute mean and count (exclude sum to reduce size)
                agg_result = df_subset.groupby(cat_col)[numeric_cols].agg(['mean', 'count']).round(2).to_dict()
                aggregates[cat_col] = {
                    "top_10_categories": agg_result,
                    "total_categories": df[cat_col].nunique(),
                    "note": f"Showing top 10 out of {df[cat_col].nunique()} categories"
                }
        
        results["compute_aggregates"] = aggregates if aggregates else {
            "message": "No categorical and numeric columns available for aggregation"
        }
        
        return {
            "current_step": "compute_aggregates",
            "results": results,
        }
    
    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables."""
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        relationships = {}
        
        # Compute correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr().round(3).to_dict()
            relationships["numeric_correlations"] = correlation_matrix
        
        # Compute summary statistics for categorical relationships (limit to top 5 categories to reduce size)
        if len(categorical_cols) > 1:
            relationship_summary = {}
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    ct = pd.crosstab(df[col1], df[col2])
                    # Store only shape and top values to prevent token overflow
                    relationship_summary[f"{col1}_vs_{col2}"] = {
                        "shape": list(ct.shape),
                        "top_combinations": ct.stack().nlargest(5).to_dict(),
                        "contingency_exists": True,
                    }
            relationships["categorical_relationships"] = relationship_summary if relationship_summary else {}
        
        results["analyze_relationships"] = relationships if relationships else {
            "message": "Insufficient columns for relationship analysis"
        }
        
        return {
            "current_step": "analyze_relationships",
            "results": results,
        }
    
    def _truncate_for_llm(results: dict, max_items: int = 5, max_key_len: int = 1000) -> str:
        """Serialize results for LLM, truncating large structures to prevent token overflow."""
        import json
        
        def truncate_value(val, depth=0):
            """Recursively truncate nested structures."""
            if depth > 2:  # Limit nesting depth
                return str(val)[:100]
            
            if isinstance(val, dict):
                if len(val) > max_items:
                    truncated = {k: v for i, (k, v) in enumerate(val.items()) if i < max_items}
                    truncated[f"... and {len(val) - max_items} more"] = None
                    return {k: truncate_value(v, depth + 1) for k, v in truncated.items()}
                return {k: truncate_value(v, depth + 1) for k, v in val.items()}
            
            elif isinstance(val, (list, tuple)):
                if len(val) > max_items:
                    truncated = list(val)[:max_items]
                    truncated.append(f"... and {len(val) - max_items} more items")
                    return truncated
                return [truncate_value(v, depth + 1) for v in val]
            
            elif isinstance(val, str):
                return val[:max_key_len]
            else:
                return val
        
        truncated = truncate_value(results)
        try:
            return json.dumps(truncated, indent=2, default=str)
        except:
            return str(truncated)[:2000]  # Fallback to truncated string
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": _truncate_for_llm(step_results)
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("detect_duplicates", detect_duplicates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_distributions", analyze_distributions_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("detect_outliers", detect_outliers_node)
    workflow.add_node("extract_observations_5", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_6", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_7", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "detect_duplicates")
    workflow.add_edge("detect_duplicates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_distributions")
    workflow.add_edge("analyze_distributions", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "detect_outliers")
    workflow.add_edge("detect_outliers", "extract_observations_5")
    workflow.add_edge("extract_observations_5", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_6")
    workflow.add_edge("extract_observations_6", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_7")
    workflow.add_edge("extract_observations_7", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
