import asyncio
from alm.utils.phoenix import register_phoenix
from alm.utils.rag_service import wait_for_rag_service
from alm.utils.job_monitor import monitor_other_job_async, wait_for_job_complete
import os
from pathlib import Path


def setup_data_directories():
    """
    Setup data directory structure for backend processing.
    Creates directories needed for log processing (not RAG-related).
    """
    print("\n" + "=" * 70)
    print("SETTING UP DATA DIRECTORY STRUCTURE")
    print("=" * 70)

    # Create logs directory for the training pipeline
    # The training pipeline uses "data/logs/failed" (relative to working directory /app)
    logs_dir = Path("data/logs/failed")

    # Create necessary directories for backend processing
    print("Creating directories...")
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {logs_dir}")

    print("=" * 70)


async def main():
    # Setup and initialization
    print("\n" + "=" * 70)
    print("ANSIBLE LOG MONITOR - BACKEND INITIALIZATION PIPELINE")
    print("=" * 70)

    # Get job names and namespace for monitoring
    rag_job_name = os.getenv(
        "RAG_INIT_JOB_NAME", "ansible-log-monitor-backend-rag-init"
    )
    namespace = os.getenv("NAMESPACE", os.getenv("POD_NAMESPACE", "default"))

    # Start monitoring the RAG init job in the background
    monitor_task = None
    try:
        monitor_task = asyncio.create_task(
            monitor_other_job_async(rag_job_name, namespace, check_interval=30)
        )
    except Exception as e:
        print(f"⚠ Warning: Could not start job monitoring: {e}")
        print("  Continuing without monitoring...")

    try:
        # Step 1: Setup data directories (create dirs, copy PDFs if needed)
        setup_data_directories()

        # Step 2: Run pipeline preparation in parallel with waiting for RAG init job
        # Preparation steps (init_tables, load_alerts, cluster_logs) don't need RAG init job
        # This saves time by running them while we wait for the RAG init job to complete
        print("\n" + "=" * 70)
        print("PREPARING PIPELINE (in parallel with RAG init job)")
        print("=" * 70)

        # Start waiting for RAG init job in background
        rag_job_wait_task = asyncio.create_task(
            wait_for_job_complete(rag_job_name, namespace, max_wait_time=600)
        )

        # Run pipeline preparation steps that don't need RAG init job
        # This includes: init_tables, load_alerts, cluster_logs (sentence transformer loading)
        from alm.pipeline.offline import training_pipeline_prepare

        alerts, cluster_labels, unique_cluster = await training_pipeline_prepare()

        print("\n" + "=" * 70)
        print("PIPELINE PREPARATION COMPLETE")
        print("  Waiting for RAG init job to complete...")
        print("=" * 70)

        # Now wait for RAG init job (may already be complete)
        # This ensures the RAG index is saved to MinIO before the RAG service tries to load it
        try:
            await rag_job_wait_task
        except (TimeoutError, RuntimeError) as e:
            print(f"\n✗ ERROR: {e}")
            print("  Cannot proceed without RAG index. Exiting...")
            raise

        # Step 3: Wait for RAG service to be ready (required for alert processing)
        # The RAG service will start after its init container detects the index in MinIO
        rag_service_url = os.getenv("RAG_SERVICE_URL", "http://alm-rag:8002")
        await wait_for_rag_service(rag_service_url)

        # Step 4: Process alerts (this requires RAG service)
        print("\n" + "=" * 70)
        print("PROCESSING ALERTS (requires RAG service)")
        print("=" * 70)

        from alm.pipeline.offline import training_pipeline_process

        await training_pipeline_process(alerts, cluster_labels, unique_cluster)

        print("\n" + "=" * 70)
        print("✓ BACKEND INITIALIZATION COMPLETE")
        print("=" * 70)

    finally:
        # Cancel monitoring task
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    register_phoenix()
    asyncio.run(main())
