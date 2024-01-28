"""This script follows from https://www.youtube.com/watch?v=g_j6ILT-X0k"""

from email.message import EmailMessage
import json
import os
import platform
import smtplib
import ssl
import time

if os.path.isfile("gmail.json"):
    with open("gmail.json", "r") as json_file:
        gmail_json = json.load(json_file)
        email_sender = gmail_json["from"]
        email_receiver = gmail_json["to"]
        email_passwd = gmail_json["passwd"]


def dict_to_table_str(dictionary):
    """Turn dictionary to string in table format."""

    table_str = str(dictionary)

    # Remove "{" and "}".
    table_str = table_str[1:-1]

    # Add "  " spaces for each row.
    table_str = table_str.split(", ")
    for i in range(len(table_str)):
        table_str[i] = "  " + table_str[i]
    table_str = "\n".join(table_str)

    return table_str


def send_email(subject, message, config):
    """Send email via gmail."""

    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_receiver

    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    em["Subject"] = f"[QML {current_time}] {subject}"

    body = "***** System *****\n"
    body += dict_to_table_str(platform.uname()._asdict())
    body += "\n\n\n"

    body += "***** Message *****\n"
    body += message
    body += "\n\n\n"

    body += "***** Configuration *****\n"
    body += dict_to_table_str(config)
    body += "\n\n\n"

    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=context) as smtp:
        smtp.login(email_sender, email_passwd)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
