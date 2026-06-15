from evaluation.deepeval_eval import evaluate_response

# print(
#     evaluate_response(
#         "What is 2+2?",
#         "The answer is 4.",
#         "4"
#     )
# )


print(
    evaluate_response(
        "Who is covered by medical insurance?",
        "Employees and their dependents are covered.",
        "employees and dependents"
    )
)