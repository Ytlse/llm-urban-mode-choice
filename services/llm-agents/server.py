import uvicorn
from backup_helper import backup_file_if_exists
from loguru import logger
from helper import setup_logging
from settings import settings

if __name__ == "__main__":
    # Backup important files
    backup_file_if_exists(settings.app.agent_memory_events_jsonl)

    # Set up logging
    setup_logging(settings)

    logger.info(f"---- Starting server ... ----")

    uvicorn.run(
        "llm_gateway.main:app",
        host=settings.server.http_host, 
        port=settings.server.http_port, 
        http="h11", 
        reload=False,
        reload_delay=0.25,
        timeout_keep_alive=5,
    )
