"""WTForms definitions for the screening demonstration."""

from flask_wtf import FlaskForm
from wtforms import FloatField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, Length, NumberRange, Optional


class ServiceForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired(), Length(max=100)])
    email = StringField("Email", validators=[Optional(), Email(), Length(max=254)])
    postalcode = StringField(
        "Postal Code / ZIP", validators=[DataRequired(), Length(min=3, max=12)]
    )
    radius = FloatField(
        "Mean radius",
        validators=[DataRequired(), NumberRange(min=6.981, max=28.11)],
    )
    texture = FloatField(
        "Mean texture",
        validators=[DataRequired(), NumberRange(min=9.71, max=39.28)],
    )
    perimeter = FloatField(
        "Mean perimeter",
        validators=[DataRequired(), NumberRange(min=43.79, max=188.5)],
    )
    submit = SubmitField("Analyze")
