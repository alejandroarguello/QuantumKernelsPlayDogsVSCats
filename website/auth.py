"""
This module contains functions related to user authentication.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user
import smtplib
from email.message import EmailMessage
import random
import string
from dotenv import load_dotenv
load_dotenv()  # This will load variables from .env into os.environ


# Create a blueprint for authentication routes
auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """
    Logs in a user if the submitted credentials are correct; otherwise, flashes an error.
    """
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)


@auth.route('/logout')
@login_required
def logout():
    """
    Logs out the current user.
    """
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    """
    Registers a new user if the provided form data is valid.
    Otherwise, flashes an appropriate error message.
    """
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        
        # Check if the email or username already exists
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', category='error')
        elif User.query.filter_by(username=username).first():
            flash('User name already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(username) < 2:
            flash('User name must be greater than 1 character.', category='error')
        elif len(username) > 20:
            flash('User name must be shorter than 21 characters.', category='error')
        elif password1 != password2:
            flash("Passwords don't match.", category='error')
        elif len(password1) < 4:
            flash('Password must be at least 4 characters.', category='error')
        else:
            # Create new user and add to database
            new_user = User(
                email=email,
                username=username,
                password=generate_password_hash(password1),
                fullname="Not specified",
                numModelsSaved="0",
                numTrainingsHistory="0"
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Account created!', category='success')
            return redirect(url_for('auth.login'))

    return render_template("sign_up.html", user=current_user)


import os
from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash
from . import db
from flask_login import current_user
import smtplib
from email.message import EmailMessage
import random
import string


@auth.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    """
    Resets the user's password if the provided email exists.
    Sends the new password via email; otherwise, flashes an error.
    """
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()

        if user:
            # Generate a random 8-character password
            source = string.ascii_letters + string.digits
            generated_password = ''.join(random.choice(source) for _ in range(8))
            
            # Hash and update the user's password
            hashed_password = generate_password_hash(generated_password)
            user.password = hashed_password
            db.session.commit()
            
            # Prepare email content
            message = (
                "Hello, as you requested, a new password has been generated for your QKPDVSC account.\n\n"
                f"Your new password is: {generated_password}\n"
                "Feel free to change your password from your profile if you feel it is convenient.\n\n"
                "Best regards, the administrator."
            )
            em = EmailMessage()
            em['From'] = os.environ.get('EMAIL_ADDRESS')
            em['To'] = email
            em['Subject'] = "New password for your QKPDVSC account"
            em.set_content(message)

            # Send the email using SMTP with credentials from the environment
            smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.environ.get('SMTP_PORT', 587))
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(os.environ.get('EMAIL_ADDRESS'), os.environ.get('EMAIL_PASSWORD'))
            server.sendmail(os.environ.get('EMAIL_ADDRESS'), email, em.as_string())
            server.quit()
            
            flash("An email has been sent to this address with the new password.", category='success')
        else:
            flash("There is no user registered with this email address. Try again.", category='error')

    return render_template("forgot_password.html", user=current_user)