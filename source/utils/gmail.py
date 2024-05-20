"""Module related to sending gmails.

This script follows from https://www.youtube.com/watch?v=g_j6ILT-X0k
"""

from email.message import EmailMessage
import os
import platform
import smtplib
import ssl
import time
import yaml

from source.utils.path import root_path

configs_dir = os.path.join(root_path, 'configs')
gmail_config_path = os.path.join(configs_dir, 'gmail.yaml')

if os.path.isfile(gmail_config_path):
    with open(gmail_config_path, 'r') as file:
        gmail_config = yaml.safe_load(file)
        email_sender = gmail_config['from']
        email_receiver = gmail_config['to']
        email_passwd = gmail_config['passwd']

def dict_to_table_str(dictionary):
    """Turn dictionary to string in table format."""

    table_str = ''
    for key, value in dictionary.items():
        table_str += f" - {key}: {value}\n"
    return table_str

def send_email(subject, message, config):
    """Send email via gmail."""

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver

    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    em['Subject'] = f"[QML {current_time}] {subject}"

    body = '***** System *****\n'
    body += dict_to_table_str(platform.uname()._asdict())
    body += '\n\n\n'

    body += '***** Message *****\n'
    body += message
    body += '\n\n\n'

    body += '***** Configuration *****\n'
    body += dict_to_table_str(config)
    body += '\n\n\n'

    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', port=465, context=context) as smtp:
        smtp.login(email_sender, email_passwd)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
