from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, FloatField


class ServiceForm(FlaskForm):
    name = StringField("Name")
    email = StringField("E-Mail")
    postalcode = StringField("Postal Code")
    radius = FloatField("Radius")
    texture = FloatField("Texture")
    perimeter = FloatField("Perimeter")
    submit = SubmitField("Submit")
