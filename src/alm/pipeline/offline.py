import asyncio
import time
from typing import List, Dict, Tuple
from alm.database import get_session
from alm.agents.graph import graph_without_clustering
from alm.agents.node import train_embed_and_cluster_logs
from alm.models import GrafanaAlert, LogEntry
from alm.database import convert_state_to_grafana_alert
from alm.database import init_tables
from alm.utils.logger import get_logger
from alm.ingestion.loki_database import LokiDataLoader
from alm.agents.state import GrafanaAlertState

logger = get_logger(__name__)


async def _add_or_update_alert(alert):
    async with get_session() as db:
        db.add(alert)
        await db.commit()
        await db.refresh(alert)


def cluster_logs(
    log_entries: List[LogEntry],
) -> Tuple[List[str], Dict[str, LogEntry]]:
    """Cluster logs and return unique alerts per cluster."""
    messages = [log_entry.message for log_entry in log_entries]
    cluster_labels = train_embed_and_cluster_logs(messages)
    unique_cluster = {
        label: log_entry for log_entry, label in zip(log_entries, cluster_labels)
    }

    logger.info(f"found {len(unique_cluster)} log templates")
    return cluster_labels, unique_cluster


async def load_log_entries():
    log_entries = await LokiDataLoader().load_and_transform()
    logger.info("log entries loaded from loki %d", len(log_entries))
    return log_entries


async def _process_alert(label: str, log_entry: LogEntry) -> Tuple[str, GrafanaAlert]:
    """Process a single alert through the graph without clustering and return (label, result)."""
    state = GrafanaAlertState(log_entry=log_entry, logCluster=label)
    result_state = await graph_without_clustering().ainvoke(state)
    return label, convert_state_to_grafana_alert(GrafanaAlertState(**result_state))


async def training_pipeline_prepare(restart_db=True, load_alerts_from_db=False):
    """
    Prepare pipeline by running steps that don't require RAG service.
    This includes database initialization, loading alerts, and clustering logs.

    Returns:
        Tuple of (log_entries, cluster_labels, unique_cluster) for later processing.
    """
    if restart_db:
        await init_tables(delete_tables=True)

    # Load log entries
    log_entries = await load_log_entries()

    # Cluster logs (loads sentence transformer model and generates embeddings)
    cluster_labels, unique_cluster = cluster_logs(log_entries)

    return log_entries, cluster_labels, unique_cluster


async def training_pipeline_process(
    log_entries: List[LogEntry],
    cluster_labels: List[str],
    unique_cluster: Dict[str, LogEntry],
):
    """
    Process alerts through the graph (requires RAG service to be ready).
    This step uses the RAG service for retrieving context from the knowledge base.

    Args:
        log_entries: List of all log entries to process
        cluster_labels: Cluster labels for each log entry
        unique_cluster: Dictionary mapping cluster labels to unique log entries
    """
    # Process all unique cluster alerts in parallel
    results = await asyncio.gather(
        *[
            _process_alert(label, log_entry)
            for label, log_entry in unique_cluster.items()
        ]
    )
    updated_alerts: Dict[str, GrafanaAlert] = dict(results)

    alerts = []
    # update alerts fields by label
    for label, log_entry in zip(cluster_labels, log_entries):
        candidate_alert = updated_alerts[label]
        # All the intermediate steps of the agent
        alert = GrafanaAlert(**candidate_alert.model_dump())
        alert.logTimestamp = log_entry.timestamp
        alert.logMessage = log_entry.message
        alert.log_labels = log_entry.log_labels.model_dump(mode="json")
        alerts.append(alert)
    # update database
    start_time = time.time()
    await asyncio.gather(*[_add_or_update_alert(alert) for alert in alerts])
    elapsed_time = time.time() - start_time
    logger.info("database alerts added - Time: %.2fs", elapsed_time)


async def training_pipeline(restart_db=True, load_alerts_from_db=False):
    """
    Run the complete training pipeline.

    This function maintains backward compatibility by calling both preparation
    and processing steps in sequence. For optimized parallel execution with
    RAG service startup, use training_pipeline_prepare() and training_pipeline_process()
    separately.
    """
    log_entries, cluster_labels, unique_cluster = await training_pipeline_prepare(
        restart_db, load_alerts_from_db
    )
    await training_pipeline_process(log_entries, cluster_labels, unique_cluster)
