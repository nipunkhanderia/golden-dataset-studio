from observability.langfuse_setup import langfuse
import inspect

print(inspect.signature(langfuse.create_event))