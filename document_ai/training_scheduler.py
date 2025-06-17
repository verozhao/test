"""Automated training scheduler for Document AI."""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Set
import logging

from .incremental_training import IncrementalTrainingManager
from .models import AutomatedTrainingConfig

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Background scheduler for automated incremental training."""

    def __init__(self):
        """Initialize the training scheduler."""
        self.manager = IncrementalTrainingManager()
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start(self):
        """Start the training scheduler."""
        if self._running:
            logger.warning("Training scheduler already running")
            return
        
        self._running = True
        logger.info("Starting training scheduler")
        
        # Start the main scheduler loop
        asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """Stop the training scheduler."""
        logger.info("Stopping training scheduler")
        self._running = False
        
        # Cancel all processor tasks
        for task in self._tasks.values():
            task.cancel()
        
        self._tasks.clear()

    async def _scheduler_loop(self):
        """Main scheduler loop that manages processor-specific tasks."""
        while self._running:
            try:
                # Get all enabled training configurations
                configs = await AutomatedTrainingConfig.find({"enabled": True}).to_list()
                
                # Start tasks for new processors
                for config in configs:
                    if config.processor_id not in self._tasks:
                        logger.info(f"Starting training task for processor {config.processor_id}")
                        task = asyncio.create_task(
                            self._processor_training_loop(config.processor_id)
                        )
                        self._tasks[config.processor_id] = task
                
                # Clean up completed tasks
                completed_processors = []
                for processor_id, task in self._tasks.items():
                    if task.done():
                        completed_processors.append(processor_id)
                
                for processor_id in completed_processors:
                    del self._tasks[processor_id]
                
                # Check for disabled processors
                active_processors = {config.processor_id for config in configs}
                for processor_id in list(self._tasks.keys()):
                    if processor_id not in active_processors:
                        logger.info(f"Stopping training task for processor {processor_id}")
                        self._tasks[processor_id].cancel()
                        del self._tasks[processor_id]
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(60)

    async def _processor_training_loop(self, processor_id: str):
        """Training loop for a specific processor.

        Args:
            processor_id: Document AI processor ID.
        """
        logger.info(f"Starting processor training loop for {processor_id}")
        
        while self._running:
            try:
                # Get current configuration
                config = await AutomatedTrainingConfig.find_one({"processor_id": processor_id})
                
                if not config or not config.enabled:
                    logger.info(f"Training disabled for processor {processor_id}")
                    break
                
                # Check if training should be triggered
                logger.debug(f"Checking training conditions for processor {processor_id}")
                batch = await self.manager.run_incremental_training(processor_id)
                
                if batch:
                    logger.info(
                        f"Training triggered for processor {processor_id}: "
                        f"batch {batch.batch_id} with {len(batch.document_ids)} documents"
                    )
                    
                    # Wait for training to complete
                    await self._wait_for_training_completion(batch.batch_id)
                else:
                    logger.debug(f"No training needed for processor {processor_id}")
                
                # Wait for the configured interval
                await asyncio.sleep(config.check_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info(f"Training loop cancelled for processor {processor_id}")
                break
            except Exception as e:
                logger.error(f"Error in processor training loop {processor_id}: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _wait_for_training_completion(self, batch_id: str, timeout_hours: int = 3):
        """Wait for a training batch to complete.

        Args:
            batch_id: Training batch ID.
            timeout_hours: Maximum hours to wait.
        """
        from .models import IncrementalTrainingBatch, DocumentAIStatus
        
        start_time = datetime.now(timezone.utc)
        timeout = timedelta(hours=timeout_hours)
        
        while datetime.now(timezone.utc) - start_time < timeout:
            batch = await IncrementalTrainingBatch.find_one({"batch_id": batch_id})
            
            if not batch:
                logger.error(f"Training batch {batch_id} not found")
                break
            
            if batch.status in [
                DocumentAIStatus.DEPLOYED,
                DocumentAIStatus.TRAINED,
                DocumentAIStatus.TRAINING_FAILED,
                DocumentAIStatus.DEPLOYMENT_FAILED,
            ]:
                logger.info(f"Training batch {batch_id} completed with status {batch.status}")
                break
            
            await asyncio.sleep(60)  # Check every minute

    def get_status(self) -> Dict[str, any]:
        """Get scheduler status.

        Returns:
            Dictionary with scheduler status information.
        """
        return {
            "running": self._running,
            "active_processors": list(self._tasks.keys()),
            "task_count": len(self._tasks),
        }


# Global scheduler instance
_scheduler = None


async def get_scheduler() -> TrainingScheduler:
    """Get the global training scheduler instance.

    Returns:
        TrainingScheduler instance.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = TrainingScheduler()
    return _scheduler


async def start_scheduler():
    """Start the global training scheduler."""
    scheduler = await get_scheduler()
    await scheduler.start()


async def stop_scheduler():
    """Stop the global training scheduler."""
    scheduler = await get_scheduler()
    await scheduler.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_scheduler())
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Scheduler stopped.")
