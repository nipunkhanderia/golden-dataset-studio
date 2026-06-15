from observability.langfuse_setup import langfuse

print(type(langfuse))
print("\nMethods:\n")

for method in dir(langfuse):
    if not method.startswith("_"):
        print(method)