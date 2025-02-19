from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from os import path
from flask_login import LoginManager
from flask_dropzone import Dropzone

# Initialize SQLAlchemy (used for database interactions)
db = SQLAlchemy()

# Name of the SQLite database file
DB_NAME = "database.db"


def create_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)

    # Application Configuration
    app.config['SECRET_KEY'] = 'admin'  # Secret key for sessions (consider changing in production)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for static files
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_NAME}"  # SQLite database URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking for performance
    app.config['SESSION_COOKIE_NAME'] = "my_session_cookie"  # Custom session cookie name

    # Configure Flask-Dropzone (for file uploads)
    app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image'
    Dropzone(app)

    # Initialize the database with the app
    db.init_app(app)

    # Register Blueprints
    from .views import views
    from .auth import auth
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    # Create the database (if it doesn't exist)
    create_database(app)

    # Setup Flask-Login
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    # Import the User model for user loading
    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        """
        Given a user ID, return the corresponding User object.
        """
        return User.query.get(int(user_id))

    return app


def create_database(app):
    """
    Create the database file and all defined tables if it doesn't exist.
    """
    # Determine the full path to the database file
    db_path = os.path.join(app.root_path, DB_NAME)
    if not path.exists(db_path):
        with app.app_context():
            db.create_all()
            print('Created Database!')