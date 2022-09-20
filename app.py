from flask import Flask, session
from flask_login import LoginManager
from datetime import timedelta
from extras.models import Base, User
from extras.extensions import db
from credentials.app import secret_key
from credentials.recaptcha import recaptcha_public_key, recaptcha_private_key
from credentials.database import connection_string_database
from blueprints.authentication import authentication
from blueprints.index import index


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"mssql+pyodbc:///?odbc_connect={connection_string_database}"
    app.config["SECRET_KEY"] = secret_key
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB max-limit.
    app.config["RECAPTCHA_PUBLIC_KEY"] = recaptcha_public_key
    app.config["RECAPTCHA_PRIVATE_KEY"] = recaptcha_private_key
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True

    db.app = app
    db.init_app(app)

    Base.prepare(db.engine, reflect=True)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.session_protection = "strong"
    login_manager.login_view = "authentication.login"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.query(User).get(int(user_id))

    app.register_blueprint(authentication)
    app.register_blueprint(index)

    @app.before_first_request
    def make_session_permanent():
        session.permanent = True
        app.permanent_session_lifetime = timedelta(seconds=120)

    return app


app = create_app()
app.app_context().push()
app.run(host="0.0.0.0", port=80)
