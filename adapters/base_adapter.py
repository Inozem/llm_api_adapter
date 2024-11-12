from abc import ABC, abstractmethod
import importlib
from types import ModuleType
from typing import Dict, Optional
import warnings

from pydantic import BaseModel, constr, model_validator, PrivateAttr

from ..errors.llm_api_error import LLMAPIError, LLMLibraryNotInstalledError
from ..responses.chat_response import ChatResponse


class LLMAdapterBase(ABC, BaseModel):
    model: str
    api_key: constr(min_length=1)
    _library: Optional[ModuleType] = PrivateAttr(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.install_library()

    @model_validator(mode='after')
    def validate(cls, values: Dict) -> Dict:
        model = values.model
        verified_models = values.verified_models
        if model not in verified_models:
            warnings.warn(
                (f"Model '{model}' is not verified for this adapter. "
                 "Continuing with the selected adapter."),
                UserWarning
            )
        return values

    @abstractmethod
    def generate_chat_answer(**kwargs) -> ChatResponse:
        """
        Generates a response based on the provided conversation.
        """
        pass

    def install_library(self) -> None:
        """
        Tries to import the required library specified by `self.library_name`.
        """
        try:
            self._library = importlib.import_module(self.library_name)
        except ModuleNotFoundError:
            raise LLMLibraryNotInstalledError(
                self.library_name, self.library_install_name
            )

    def handle_error(cls, e: Exception, company: str) -> None:
        for error_class in LLMAPIError.__subclasses__():
            api_errors = getattr(error_class, f"{company}_api_errors")
            if type(e).__name__ in api_errors:
                raise error_class(f"{error_class.message} {str(e)}")
        raise LLMAPIError(f"{LLMAPIError.message} {str(e)}")
