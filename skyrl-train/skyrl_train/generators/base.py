from typing import List, Dict, Any, TypedDict, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from skyrl_train.inference_engines.base import ConversationType


@dataclass
class TrajectoryID:
    row_id: str  # Unique identifier from dataset (instance_id, prompt_id, etc.)
    repetition_id: int  # Which sample/repetition for this UID (0, 1, 2... for GRPO)


class GeneratorInput(TypedDict):
    prompts: List[ConversationType]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[TrajectoryID]]


class GeneratorOutput(TypedDict):
    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    rollout_logprobs: Optional[List[List[float]]]


class GeneratorInterface(ABC):
    @abstractmethod
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        raise NotImplementedError()
