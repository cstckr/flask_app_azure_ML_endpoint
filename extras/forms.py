from flask_wtf import FlaskForm, RecaptchaField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField, PasswordField
from wtforms.validators import InputRequired, Length, EqualTo


class UploadForm(FlaskForm):
    file = FileField(validators=[FileRequired(), FileAllowed(["jpg"])])
    submit = SubmitField("Upload")


class LoginForm(FlaskForm):
    recaptcha = RecaptchaField()
    username = StringField(validators=[InputRequired(), Length(min=4, max=30)],
                           render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4,
                                                                 max=30)],
                             render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")


class ChangePasswordForm(FlaskForm):
    old_password = PasswordField(validators=[InputRequired()],
                                 render_kw={"placeholder": "Old password"})
    new_password1 = PasswordField(validators=[
        InputRequired(),
        Length(min=4, max=30,
               message="Password must contain at least 4 characters")],
        render_kw={"placeholder": "New password"})
    new_password2 = PasswordField(validators=[
        InputRequired(),
        EqualTo("new_password1", message="New passwords must match")],
        render_kw={"placeholder": "Repeat new password"})
    submit = SubmitField("Change")
