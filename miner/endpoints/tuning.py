import os
from datetime import datetime
from datetime import timedelta

import toml
import yaml
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_get_request
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
# from miner.config import WorkerConfig
# from miner.dependencies import get_worker_config
# from miner.logic.job_handler import create_job_diffusion
# from miner.logic.job_handler import create_job_text
import requests


logger = get_logger(__name__)
def setup_logging():
    """
    Configure logging to output to both console and file.
    """
    import logging
    import sys
    from logging.handlers import RotatingFileHandler
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicate logs
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/miner_tuning.log", 
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

# Initialize logging
setup_logging()



current_job_finish_time = None
ENDPOINT = os.getenv("ENDPOINT", "http://localhost:9999")

async def tune_model_text(
    train_request: TrainRequestText,
):
    logger.info(f"train_request: {train_request.model_dump()}")

    try:
        url = f"{ENDPOINT}/train"
        data = {
            "train_request": train_request.model_dump(),
        }
        response = requests.post(url, json=data).json()
        logger.info(f"my response for tune_model_text {train_request.task_id} is '{response}'")
        return {"message": "Training job enqueued.", "task_id": train_request.task_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def tune_model_diffusion(
    train_request: TrainRequestImage,
):
    return {"message": "Training job enqueued.", "task_id": "xxx"}


async def get_latest_model_submission(task_id: str) -> str:
    logger.info(f"get_latest_model_submission: {task_id}")
    try:
        params = {"miner_uid": -1, "task_id": task_id}
        response = requests.post(f"{ENDPOINT}/submision", params=params).json()
        logger.info(f"my response for {task_id} is '{response}'")
        return response["result"]
    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskOffer,
) -> MinerTaskResponse:
    logger.info(f"task_offer: {request.model_dump()}")
    try:
        if request.task_type not in [TaskType.INSTRUCTTEXTTASK]:
            logger.info(f"task type is not instruct text task")
            return MinerTaskResponse(
                message="I only accept text tasks",
                accepted=False,
            )
            
        url = f"{ENDPOINT}/acceptance"
        data = request.model_dump()
        response = requests.post(url, json=data).json()
        logger.info(f"my response for {request.task_id} is '{response}'")
        return MinerTaskResponse(
            message=response["error"] if response["error"] else "I can handle this model",
            accepted=response["accepted"],
        )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
) -> MinerTaskResponse:
    logger.info(f"task_offer_image: {request.model_dump()}")
    try:
        return MinerTaskResponse(
            message="I only accept text tasks",
            accepted=False,
        )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )
    router.add_api_route(
        "/start_training/",  # TODO: change to /start_training_text or similar
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_image/",
        tune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    return router
