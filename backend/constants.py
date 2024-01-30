import os
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.prompts import PromptTemplate

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
if PERSIST_DIRECTORY is None:
    raise Exception("Please set the PERSIST_DIRECTORY environment variable")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)



DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""[INST] <>{system_prompt}<>{prompt} [/INST]""".strip()


SYSTEM_PROMPT = "You are a question and answer bot. You are given information from a search engine and should only answer based from the following search result to the question at the end you can rephrase or summrize. If you don't know the answer, just say that you don't know, don't EVER try to make up an answer. you're bad at ansewring anything about csv, dataset, datasets or tsv, you're also bad at math"

template = generate_prompt("""{context}Question: {question}""",system_prompt=SYSTEM_PROMPT,)

QA_CHAIN_PROMPT = PromptTemplate(input_variables=['context','question'],template=template,)



# import os
# from dotenv import load_dotenv
# from chromadb.config import Settings
# from langchain.prompts import PromptTemplate
#
# load_dotenv()
#
# # Define the folder for storing database
# PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
# if PERSIST_DIRECTORY is None:
#     raise Exception("Please set the PERSIST_DIRECTORY environment variable")
#
# # Define the Chroma settings
# CHROMA_SETTINGS = Settings(
#         persist_directory=PERSIST_DIRECTORY,
#         anonymized_telemetry=False
# )
#
#
#
# DEFAULT_SYSTEM_PROMPT = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """.strip()
#
#
# def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
#     return f"""[INST] <>{system_prompt}<>{prompt} [/INST]""".strip()
#
#
# SYSTEM_PROMPT = f""""You are a question and answer bot. You are given information from a search engine and should only answer based from the following search result to the question at the end you can rephrase or summrize. If you don't know the answer, just say that you don't know, don't EVER try to make up an answer." \
#                 "You are an AI language model tasked with analyzing a corpus of text data related to an electric company. Your goal is to identify and categorize text passages as either sensitive or non-sensitive data. The electric company operates in a regulated industry, and it is critical to ensure the confidentiality of sensitive information to protect customer privacy, comply with data protection regulations, and maintain the security of its infrastructure.
# Sensitive Data Examples (25):
#
# 1. Extract: "John Doe's home address is 123 Main Street."
# 2. Extract: "Customer ID: 987654321"
# 3. Extract: "Mary Smith's phone number is (555) 555-5555."
# 4. Extract: "Social Security Number: 123-45-6789"
# 5. Extract: "Customer Account Number: 543210"
# 6. Extract: "Credit Card Number: 1234-5678-9012-3456"
# 7. Extract: "Bank Account Information: ACCT-987654321"
# 8. Extract: "Employee ID: E12345"
# 9. Extract: "Jane Brown's salary is $60,000 per year."
# 10. Extract: "Repair Log - Grid Component Serial Number: XYZ12345"
# 11. Extract: "Maintenance Password: GridSecure123"
# 12. Extract: "Energy Consumption on 5/25/2023: 850 kWh"
# 13. Extract: "Utility Bill - Account Balance: $500.00"
# 14. Extract: "Invoice Due Date: 06/15/2023"
# 15. Extract: "Customer Email: customer@example.com"
# 16. Extract: "Tax ID: 12-3456789"
# 17. Extract: "Employee Address: 456 Elm Street"
# 18. Extract: "Account Password: SecretP@ssword1"
# 19. Extract: "Smart Meter Reading: 1530 kWh"
# 20. Extract: "Grid Access Code: GridAccess1234"
# 21. Extract: "Energy Usage Data - 123 Main St, April 2023: 500 kWh"
# 22. Extract: "Repair Report - Location: Substation A"
# 23. Extract: "Customer Banking Details: Bank X, ACCT-54321"
# 24. Extract: "Employee Tax Forms: W-2, 2022"
# 25. Extract: "Security Clearance Level: Level 3"
#
# Non-Sensitive Data Examples (25):
#
# 1. Extract: "Electric Company History: Founded in 1980."
# 2. Extract: "Announcement: Join us for our annual energy-saving event."
# 3. Extract: "Public Service Message: Conserve energy during peak hours."
# 4. Extract: "Annual Report 2022: Financial highlights."
# 5. Extract: "Grid Performance Report: No outages reported today."
# 6. Extract: "Grid Diagram: General grid layout."
# 7. Extract: "Service Upgrade Notice: Scheduled for 6/10/2023."
# 8. Extract: "Sustainability Report: Environmental initiatives."
# 9. Extract: "Energy Savings Tips: Lower your energy bill."
# 10. Extract: "Public Contact Information: Contact us at info@electricco.com."
# 11. Extract: "Public Event: Electric Vehicle Charging Workshop."
# 12. Extract: "Grid Performance Data: Daily power generation statistics."
# 13. Extract: "Non-Identifiable Usage Data: Average energy usage per household."
# 14. Extract: "Customer Feedback: 'Great service, thank you!'"
# 15. Extract: "Outage Alert: Power restored in your area."
# 16. Extract: "Industry News: Renewable energy trends."
# 17. Extract: "Grid Maintenance Notice: Substation B maintenance."
# 18. Extract: "Grid Component Specs: Technical specifications."
# 19. Extract: "Public Survey: Rate your energy service."
# 20. Extract: "Energy Efficiency Tips: Upgrade your appliances."
# 21. Extract: "Public Announcement: New office location."
# 22. Extract: "Employee Recognition: Employee of the Month - John Smith."
# 23. Extract: "Public Safety Guidelines: Stay safe around power lines."
# 24. Extract: "Market Trends: Electricity prices in your region."
# 25. Extract: "Energy Saving Challenge: Win a free energy audit."
#
# Your task is to examine text passages and label them as "Sensitive" or "Non-Sensitive" based on the examples provided. Ensure that sensitive data is answed by you for now write "sens data" and handled with the utmost care and security, and non-sensitive data can be sent to chatgpt if it can write 'i'll send this to chatgpt'"""
#
#
# template = generate_prompt("""{context}Question: {question}""",system_prompt=SYSTEM_PROMPT,)
#
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=['context','question'],template=template,)