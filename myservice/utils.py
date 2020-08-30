# -*- coding: utf-8 -*-
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

import myservice.settings as settings

def send_email(result, recipients):
    email_my_sender = settings.EMAIL_HOST_USER
    email_my_pass = settings.EMAIL_HOST_PASSWORD

    status = True
    try:
        message = MIMEMultipart()
        message['From'] = formataddr(['LBCI', email_my_sender])
        message['To'] = formataddr(['Friend', ''])
        message['Subject'] = 'Result of DeepAVP'
        message.attach(MIMEText('This is the result of DeepAVP', 'plain', 'utf-8'))

        att1 = MIMEText(result, 'base64', 'utf-8')
        att1["Content-Type"] = 'application/octet-stream'
        att1["Content-Disposition"] = 'attachment; filename="result.txt"'
        message.attach(att1)

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(email_my_sender, email_my_pass)
        server.sendmail(email_my_sender, recipients, message.as_string())
        server.quit()
    except Exception as e:
        status = False
        print(e)
        print('Send Email Failed!')

    return status