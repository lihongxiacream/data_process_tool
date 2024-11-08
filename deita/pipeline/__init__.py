from .embed_pipeline import EmbedPipeline
from .filter_pipeline import FilterPipeline
from .base import PipelineRegistry
from typing import Callable


PipelineRegistry.register("embed_pipeline", EmbedPipeline)
PipelineRegistry.register("filter_pipeline", FilterPipeline)

class Pipeline:
    
    def __new__(cls, name, **kwargs) -> Callable:
        
        PipelineClass = PipelineRegistry.get_pipeline(name)
        return PipelineClass(name, **kwargs)