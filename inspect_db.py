# inspect_db.py
from main import app
from website import db
from website import models


def list_users():
    """Print all users."""
    users = models.User.query.all()
    for user in users:
        print(f"ID: {user.id}, Username: {user.username}, Email: {user.email}")

def get_user(user_id):
    """Print details of a user by ID."""
    user = models.User.query.get(user_id)
    if user:
        print(f"ID: {user.id}, Username: {user.username}, Email: {user.email}, Password: {user.password}")
    else:
        print("User not found.")

def create_user(email, password, username, fullname, numModelsSaved=0, numTrainingsHistory=0, profile_picture=None):
    """Create and add a new user."""
    new_user = User(email, password, username, fullname, numModelsSaved, numTrainingsHistory, profile_picture)
    db.session.add(new_user)
    db.session.commit()
    print(f"Created user {username} with email {email}.")

def update_user_email(user_id, new_email):
    """Update a user's email."""
    user = models.User.query.get(user_id)
    if user:
        user.email = new_email
        db.session.commit()
        print(f"Updated email for user {user.username} (ID: {user.id}) to {new_email}.")
    else:
        print("User not found.")

def delete_user(user_id):
    """Delete a user by ID."""
    user = models.User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        print(f"Deleted user {user.username} (ID: {user.id}).")
    else:
        print("User not found.")

def list_models():
    """Print all saved models."""
    models = models.ModelSaved.query.all()
    for model in models:
        print(f"Name: {model.name}, Date: {model.date}, Test Accuracy: {model.test_accuracy}")

def list_reports():
    """Print all PDF reports."""
    reports = models.ReportPDF.query.all()
    for report in reports:
        print(f"Name: {report.name}, Date: {report.date}, Path: {report.path}")

def main_menu():
    """Display the interactive menu and execute the chosen function."""
    while True:
        print("\nSelect an option:")
        print("1. List all users")
        print("2. Get user by ID")
        print("3. Create a new user")
        print("4. Update user email")
        print("5. Delete a user")
        print("6. List all models")
        print("7. List all reports")
        print("8. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            list_users()
        elif choice == '2':
            try:
                user_id = int(input("Enter user ID: ").strip())
                get_user(user_id)
            except ValueError:
                print("Invalid ID. Please enter a valid integer.")
        elif choice == '3':
            email = input("Enter email: ").strip()
            password = input("Enter password (should be hashed normally): ").strip()
            username = input("Enter username: ").strip()
            fullname = input("Enter full name: ").strip()
            create_user(email, password, username, fullname)
        elif choice == '4':
            try:
                user_id = int(input("Enter user ID: ").strip())
                new_email = input("Enter new email: ").strip()
                update_user_email(user_id, new_email)
            except ValueError:
                print("Invalid ID. Please enter a valid integer.")
        elif choice == '5':
            try:
                user_id = int(input("Enter user ID to delete: ").strip())
                delete_user(user_id)
            except ValueError:
                print("Invalid ID. Please enter a valid integer.")
        elif choice == '6':
            list_models()
        elif choice == '7':
            list_reports()
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Make sure to push an application context to perform database operations
    with app.app_context():
        main_menu()