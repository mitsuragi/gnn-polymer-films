import hashlib

from db.db_manager import get_user
from db.models import User

class AuthService:
    def __init__(self, sessionmaker):
        self.current_user = None
        self.sessionmaker = sessionmaker

    def authenticate(self, username: str, password: str) -> None | User:
        with self.sessionmaker() as session:
            user = get_user(session, username)

        print(user)

        if not user:
            return None

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if user.Password != password_hash:
            return None

        self.current_user = user

        return self.current_user

    def logout(self):
        self.current_user = None
