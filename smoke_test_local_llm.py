from local_llm import extract_key_points


transcript = """
The team will move the launch from Monday to Wednesday because the payment
integration needs another round of testing. Maya will update the schedule,
and the support team will prepare a customer notice before the new launch date.
"""

print(extract_key_points(transcript))
