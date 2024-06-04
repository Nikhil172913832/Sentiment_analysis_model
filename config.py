import os
import secrets

SECRET_KEY = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
MONGODB_URL = os.getenv('MONGODB_URL')
