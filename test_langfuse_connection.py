# test_langfuse_connection.py

from observability.langfuse_setup import langfuse

print("   Auth Check:", langfuse.auth_check())